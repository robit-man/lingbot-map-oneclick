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
response="$(curl -fsS \
  -H "Authorization: Bearer ${LINGBOT_API_TOKEN}" \
  -F "files=@${video_path};type=video/mp4" \
  -F fps=4 \
  -F max_frames=8 \
  -F num_scale_frames=2 \
  -F keyframe_interval=2 \
  -F confidence_percentile=50 \
  -F include_cameras=true \
  "${base_url}/api/jobs")"
job_id="$(printf '%s' "${response}" | "${python_bin}" -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
printf 'smoke: job %s queued\n' "${job_id}"

last_message=""
while (( SECONDS < deadline )); do
  response="$(curl -fsS \
    -H "Authorization: Bearer ${LINGBOT_API_TOKEN}" \
    "${base_url}/api/jobs/${job_id}")"
  IFS=$'\t' read -r status progress message < <(
    printf '%s' "${response}" | "${python_bin}" -c '
import json
import sys
job = json.load(sys.stdin)
message = str(job.get("message", "")).replace("\t", " ").replace("\n", " ")
print(job["status"], job.get("progress", 0), message, sep="\t")
'
  )
  if [[ "${message}" != "${last_message}" ]]; then
    printf 'smoke: %3s%% %s\n' "${progress}" "${message}"
    last_message="${message}"
  fi
  if [[ "${status}" == "complete" ]]; then
    result_url="$(printf '%s' "${response}" | "${python_bin}" -c 'import json,sys; print(json.load(sys.stdin)["result_url"])')"
    curl -fsS \
      -H "Authorization: Bearer ${LINGBOT_API_TOKEN}" \
      "${base_url}${result_url}" \
      -o "${result_path}"
    [[ "$(head -c 4 "${result_path}")" == "glTF" ]] || {
      printf 'error: reconstruction is not a binary GLB\n' >&2
      exit 1
    }
    geometry_count="$(docker run --rm \
      --entrypoint python \
      -v "${result_path}:/result.glb:ro" \
      lingbot-map-web:local \
      -c 'import trimesh; scene=trimesh.load("/result.glb"); count=len(scene.geometry); assert count; print(count)')"
    printf 'smoke: PASS — browser-ready GLB is %s bytes with %s geometries\n' \
      "$(stat -c %s "${result_path}")" "${geometry_count}"
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
