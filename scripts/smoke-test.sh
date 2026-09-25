#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
env_file="${project_dir}/.env"
python_bin="${LINGBOT_BOOTSTRAP_PYTHON:-${project_dir}/.venv/bin/python}"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

if [[ ! -f "${env_file}" ]]; then
  printf 'error: run ./run.sh before the smoke test\n' >&2
  exit 1
fi
if [[ ! -x "${python_bin}" ]]; then
  printf 'error: bootstrap venv is missing; run ./run.sh first\n' >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a

base_url="${1:-http://127.0.0.1:${LINGBOT_PORT:-8080}}"
video_path="${temporary_dir}/walkaround.mp4"
result_path="${temporary_dir}/reconstruction.glb"
point_cloud_path="${temporary_dir}/point-cloud-lod.ply"
deadline=$((SECONDS + ${LINGBOT_SMOKE_TIMEOUT:-1800}))

curl -fsS "${base_url}/readyz" >/dev/null
printf 'smoke: encoding a real MP4 from the bundled walkaround\n'
docker run --rm \
  --entrypoint ffmpeg \
  -v "${project_dir}/example/loop:/input:ro" \
  -v "${temporary_dir}:/output" \
  lingbot-map-web:local \
  -hide_banner -loglevel error \
  -framerate 4 -pattern_type glob -i '/input/*.png' \
  -frames:v 12 -vf 'scale=iw-mod(iw\,2):ih-mod(ih\,2)' \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart -y /output/walkaround.mp4

printf 'smoke: uploading video (%s bytes)\n' "$(stat -c %s "${video_path}")"
manifest="$("${python_bin}" -c '
import json
print(json.dumps({
  "schema": "noclip.lingbot.request/1.0",
  "captureSessionId": "smoke-contract",
  "coordinateSystem": {
    "geodetic": "WGS84", "localFrame": "ENU",
    "cameraAxes": "opencv_x_right_y_down_z_forward",
    "quaternionOrder": "xyzw", "units": "meters"
  },
  "media": [], "poses": [],
  "options": {"videoFps": 4, "maxFrames": 8, "numScaleFrames": 2,
              "keyframeInterval": 2, "confidencePercentile": 50,
              "includeCameras": False}
}))
')"
response="$(curl -fsS \
  -H "Authorization: Bearer ${LINGBOT_API_TOKEN}" \
  -H "Idempotency-Key: smoke-$(date +%s)" \
  -F "media=@${video_path};type=video/mp4" \
  -F "manifest=${manifest}" \
  "${base_url}/v1/reconstructions")"
job_id="$(printf '%s' "${response}" | "${python_bin}" -c 'import json,sys; print(json.load(sys.stdin)["jobId"])')"
printf 'smoke: job %s queued\n' "${job_id}"

last_message=""
while (( SECONDS < deadline )); do
  response="$(curl -fsS \
    -H "Authorization: Bearer ${LINGBOT_API_TOKEN}" \
    "${base_url}/v1/reconstructions/${job_id}")"
  IFS=$'\t' read -r status progress message < <(
    printf '%s' "${response}" | "${python_bin}" -c '
import json
import sys
job = json.load(sys.stdin)
message = str(job.get("message", "")).replace("\t", " ").replace("\n", " ")
print(job["status"], round(float(job.get("progress", 0)) * 100), message, sep="\t")
'
  )
  if [[ "${message}" != "${last_message}" ]]; then
    printf 'smoke: %3s%% %s\n' "${progress}" "${message}"
    last_message="${message}"
  fi
  if [[ "${status}" == "completed" ]]; then
    response="$(curl -fsS \
      -H "Authorization: Bearer ${LINGBOT_API_TOKEN}" \
      "${base_url}/v1/reconstructions/${job_id}/result")"
    readarray -t artifact_urls < <(printf '%s' "${response}" | "${python_bin}" -c '
import json,sys
artifacts = {item["kind"]: item["url"] for item in json.load(sys.stdin)["artifacts"]}
required = {"reconstruction_glb", "point_cloud", "trajectory", "manifest", "confidence"}
assert required <= artifacts.keys(), sorted(artifacts)
print(artifacts["reconstruction_glb"])
print(artifacts["point_cloud"])
')
    curl -fsS \
      -H "Authorization: Bearer ${LINGBOT_API_TOKEN}" \
      "${base_url}${artifact_urls[0]}" \
      -o "${result_path}"
    curl -fsS \
      -H "Authorization: Bearer ${LINGBOT_API_TOKEN}" \
      "${base_url}${artifact_urls[1]}" \
      -o "${point_cloud_path}"
    [[ "$(head -c 4 "${result_path}")" == "glTF" ]] || {
      printf 'error: reconstruction is not a binary GLB\n' >&2
      exit 1
    }
    [[ "$(head -c 3 "${point_cloud_path}")" == "ply" ]] || {
      printf 'error: point-cloud LOD is not a PLY artifact\n' >&2
      exit 1
    }
    geometry_count="$(docker run --rm \
      --entrypoint python \
      -v "${result_path}:/result.glb:ro" \
      lingbot-map-web:local \
      -c 'import trimesh; scene=trimesh.load("/result.glb"); count=len(scene.geometry); assert count; print(count)')"
    printf 'smoke: PASS — GLB is %s bytes with %s geometries; PLY LOD is %s bytes\n' \
      "$(stat -c %s "${result_path}")" "${geometry_count}" \
      "$(stat -c %s "${point_cloud_path}")"
    exit 0
  fi
  if [[ "${status}" == "failed" ]]; then
    printf 'error: %s\n' "${message}" >&2
    exit 1
  fi
  sleep 2
done

printf 'error: smoke job timed out\n' >&2
exit 1
