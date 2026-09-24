#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
env_file="${project_dir}/.env"
compose_file="${project_dir}/compose.yaml"
service_name="lingbot-map.service"

usage() {
  cat <<'USAGE'
Usage: scripts/deploy.sh COMMAND

Commands:
  run              Build, pull/verify weights, acquire a scoped GPU, and run
  build            Build the application image and minified frontend
  pull-weights     Pull and verify the configured Hugging Face checkpoint
  install-service  Install and start the supervised systemd user service
  down             Stop containers (model/job data is preserved)
  status           Show the systemd, Compose, and GPU-broker state
  logs             Follow application and tunnel logs
  url              Print the active Quick Tunnel or configured public URL
  token            Print the generated browser/API token
  verify           Check model integrity, app readiness, and tunnel state
USAGE
}

ensure_env() {
  if [[ ! -f "${env_file}" ]]; then
    umask 077
    cp "${project_dir}/.env.example" "${env_file}"
  fi
  if ! awk -F= '$1 == "LINGBOT_API_TOKEN" && length($2) >= 24 { found=1 } END { exit !found }' "${env_file}"; then
    local generated_token temporary_env
    generated_token="$(openssl rand -hex 24)"
    temporary_env="$(mktemp "${project_dir}/.env.XXXXXX")"
    awk -v token="${generated_token}" '
      BEGIN { replaced=0 }
      /^LINGBOT_API_TOKEN=/ && !replaced { print "LINGBOT_API_TOKEN=" token; replaced=1; next }
      { print }
      END { if (!replaced) print "LINGBOT_API_TOKEN=" token }
    ' "${env_file}" >"${temporary_env}"
    chmod 600 "${temporary_env}"
    mv "${temporary_env}" "${env_file}"
  fi
  chmod 600 "${env_file}"
}

load_env() {
  ensure_env
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
  export LINGBOT_PORT="${LINGBOT_PORT:-8080}"
  export LINGBOT_DEPLOYMENT_MODE="${LINGBOT_DEPLOYMENT_MODE:-public}"
  export LINGBOT_VRAM_MIB="${LINGBOT_VRAM_MIB:-32768}"
  export LINGBOT_READY_TIMEOUT="${LINGBOT_READY_TIMEOUT:-1800}"
  if [[ "${LINGBOT_DEPLOYMENT_MODE}" != "public" && "${LINGBOT_DEPLOYMENT_MODE}" != "internal" ]]; then
    printf 'LINGBOT_DEPLOYMENT_MODE must be public or internal\n' >&2
    return 2
  fi
}

compose_with_placeholders() {
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-00000000-0000-0000-0000-000000000000}" \
  OLLAMA_UNIFY_GPU_LEASE="${OLLAMA_UNIFY_GPU_LEASE:-lease-placeholder}" \
    docker compose \
      --project-directory "${project_dir}" \
      --env-file "${env_file}" \
      -f "${compose_file}" \
      "$@"
}

build_image() {
  load_env
  compose_with_placeholders build app
}

pull_weights() {
  load_env
  mkdir -p "${project_dir}/data/models" "${project_dir}/data/jobs"
  local bootstrap_python
  bootstrap_python="${LINGBOT_BOOTSTRAP_PYTHON:-${project_dir}/.venv/bin/python}"
  if [[ -x "${bootstrap_python}" ]]; then
    (
      cd "${project_dir}"
      LINGBOT_MODEL_DIR="${project_dir}/data/models" \
      HF_HOME="${project_dir}/data/models/.cache/huggingface" \
        "${bootstrap_python}" -m webapp.weights
    )
  else
    compose_with_placeholders run --rm --no-deps weights
  fi
}

select_gpu() {
  local discovery_file selected
  discovery_file="$(mktemp)"
  trap 'rm -f -- "${discovery_file}"' RETURN
  docker gpu discover >"${discovery_file}"
  selected="$(python3 "${project_dir}/webapp/gpu_selection.py" \
    "${discovery_file}" \
    "${LINGBOT_GPU_UUID:-}" \
    "${LINGBOT_GPU_INDEX:-}")"
  printf '%s\n' "${selected}"
}

run_foreground() {
  load_env
  build_image
  pull_weights

  local gpu_uuid tunnel_service ready_command
  local -a services
  gpu_uuid="$(select_gpu)"
  tunnel_service=""
  if [[ "${LINGBOT_DEPLOYMENT_MODE}" == "public" ]]; then
    tunnel_service="cloudflared"
  fi
  if [[ "${LINGBOT_DEPLOYMENT_MODE}" == "public" && -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]]; then
    if [[ -z "${LINGBOT_PUBLIC_URL:-}" ]]; then
      printf 'LINGBOT_PUBLIC_URL is required with CLOUDFLARE_TUNNEL_TOKEN\n' >&2
      return 2
    fi
    tunnel_service="cloudflared-named"
  fi
  ready_command="curl -fsS http://127.0.0.1:${LINGBOT_PORT}/readyz >/dev/null"
  services=(app)
  if [[ -n "${tunnel_service}" ]]; then
    services+=("${tunnel_service}")
  fi
  printf 'LingBot-Map: reserving %s MiB on %s\n' "${LINGBOT_VRAM_MIB}" "${gpu_uuid}"

  export CUDA_VISIBLE_DEVICES="${gpu_uuid}"
  docker gpu run \
    --owner lingbot-map \
    --vram-mib "${LINGBOT_VRAM_MIB}" \
    --gpu "${gpu_uuid}" \
    --ready-timeout "${LINGBOT_READY_TIMEOUT}" \
    --ready-command "${ready_command}" \
    -- \
    docker compose \
      --project-directory "${project_dir}" \
      --env-file "${env_file}" \
      -f "${compose_file}" \
      up --remove-orphans --abort-on-container-failure "${services[@]}"
}

down_stack() {
  load_env
  compose_with_placeholders --profile named-tunnel down --remove-orphans
}

install_service() {
  load_env
  local config_root unit_dir unit_path template
  config_root="${XDG_CONFIG_HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)/.config}"
  unit_dir="${config_root}/systemd/user"
  unit_path="${unit_dir}/${service_name}"
  template="${project_dir}/deploy/systemd/lingbot-map.service.in"
  mkdir -p "${unit_dir}"
  python3 - "${template}" "${unit_path}" "${project_dir}" <<'PY'
from pathlib import Path
import sys

source, destination, workdir = map(Path, sys.argv[1:])
destination.write_text(
    source.read_text(encoding="utf-8").replace("@WORKDIR@", str(workdir)),
    encoding="utf-8",
)
PY
  systemctl --user daemon-reload
  systemctl --user enable "${service_name}"
  systemctl --user stop "${service_name}" 2>/dev/null || true
  systemctl --user reset-failed "${service_name}" 2>/dev/null || true
  systemctl --user start "${service_name}"
  printf 'Installed %s\n' "${unit_path}"
}

show_status() {
  load_env
  systemctl --user --no-pager --full status "${service_name}" || true
  compose_with_placeholders --profile named-tunnel ps
  docker gpu status
}

show_logs() {
  load_env
  compose_with_placeholders --profile named-tunnel logs --tail 200 -f app cloudflared cloudflared-named
}

show_url() {
  load_env
  if [[ "${LINGBOT_DEPLOYMENT_MODE}" == "internal" ]]; then
    printf 'http://127.0.0.1:%s\n' "${LINGBOT_PORT}"
    return
  fi
  if [[ -n "${LINGBOT_PUBLIC_URL:-}" ]]; then
    printf '%s\n' "${LINGBOT_PUBLIC_URL}"
    return
  fi
  local discovered_url
  discovered_url="$(
    compose_with_placeholders logs --no-color cloudflared 2>/dev/null \
      | grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' \
      | tail -n 1 || true
  )"
  if [[ -z "${discovered_url}" ]]; then
    printf 'No Quick Tunnel URL found yet. Check: scripts/deploy.sh logs\n' >&2
    return 1
  fi
  printf '%s\n' "${discovered_url}"
}

show_token() {
  load_env
  awk -F= '$1 == "LINGBOT_API_TOKEN" { print substr($0, index($0, "=") + 1); exit }' "${env_file}"
}

verify_deployment() {
  load_env
  printf 'weight: '
  pull_weights >/dev/null
  printf 'PASS\n'
  printf 'ready:  '
  curl -fsS "http://127.0.0.1:${LINGBOT_PORT}/readyz" >/dev/null
  printf 'PASS\n'
  printf 'tunnel: '
  if [[ "${LINGBOT_DEPLOYMENT_MODE}" == "internal" ]]; then
    printf 'SKIP (internal mode)\n'
  else
    show_url >/dev/null
    printf 'PASS\n'
  fi
  printf 'overall: PASS\n'
}

command_name="${1:-}"
case "${command_name}" in
  run) run_foreground ;;
  build) build_image ;;
  pull-weights) pull_weights ;;
  install-service) install_service ;;
  down) down_stack ;;
  status) show_status ;;
  logs) show_logs ;;
  url) show_url ;;
  token) show_token ;;
  verify) verify_deployment ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
