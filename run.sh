#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${project_dir}/.venv"
requirements_file="${project_dir}/deploy/bootstrap-requirements.txt"
requirements_stamp="${venv_dir}/.bootstrap-requirements.sha256"
deploy_script="${project_dir}/scripts/deploy.sh"
cloudflared_image="cloudflare/cloudflared:2026.8.2"

usage() {
  cat <<'USAGE'
Usage: ./run.sh [COMMAND]

Commands:
  start       Set up everything, start the supervised site, and print access details (default)
  foreground Set up everything and keep the deployment attached to this terminal
  restart     Restart the supervised deployment and wait until it is ready
  stop        Stop the service and containers without deleting models or results
  status      Show service, container, and GPU reservation state
  logs        Follow application and Cloudflared logs
  url         Print the active public URL
  token       Print the generated browser access token
  verify      Verify checkpoint integrity, model readiness, and the tunnel
  smoke       Upload a generated MP4 and verify a browser-ready 3D result
  test        Run source tests and build the minified browser app
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

ensure_host_tools() {
  require_command python3
  require_command docker
  require_command curl
  require_command openssl
  require_command sha256sum
  require_command systemctl
  docker info >/dev/null 2>&1 || die "Docker is not running or is not accessible to this user"
  docker compose version >/dev/null 2>&1 || die "Docker Compose is unavailable"
  docker gpu --help >/dev/null 2>&1 || die "the docker gpu broker command is unavailable"
}

ensure_venv() {
  local current_hash installed_hash
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    printf 'bootstrap: creating %s\n' "${venv_dir}"
    python3 -m venv "${venv_dir}" || die "could not create the Python venv (install python3-venv and retry)"
  fi

  current_hash="$(sha256sum "${requirements_file}" | awk '{print $1}')"
  installed_hash=""
  if [[ -f "${requirements_stamp}" ]]; then
    installed_hash="$(tr -d '\n' <"${requirements_stamp}")"
  fi
  if [[ "${current_hash}" != "${installed_hash}" ]]; then
    printf 'bootstrap: installing checkpoint tooling\n'
    "${venv_dir}/bin/python" -m pip install --disable-pip-version-check --upgrade pip
    "${venv_dir}/bin/python" -m pip install --disable-pip-version-check -r "${requirements_file}"
    printf '%s\n' "${current_hash}" >"${requirements_stamp}"
  fi
  export LINGBOT_BOOTSTRAP_PYTHON="${venv_dir}/bin/python"
}

ensure_cloudflared() {
  local deployment_mode
  "${deploy_script}" token >/dev/null
  deployment_mode="$(awk -F= '$1 == "LINGBOT_DEPLOYMENT_MODE" { print $2; exit }' "${project_dir}/.env")"
  if [[ "${deployment_mode:-public}" == "internal" ]]; then
    return
  fi
  if ! docker image inspect "${cloudflared_image}" >/dev/null 2>&1; then
    printf 'bootstrap: installing Cloudflared container %s\n' "${cloudflared_image}"
    docker pull "${cloudflared_image}"
  fi
}

configure_host_port() {
  local current_port selected_port temporary_env
  "${deploy_script}" token >/dev/null
  current_port="$(awk -F= '$1 == "LINGBOT_PORT" { print $2; exit }' "${project_dir}/.env")"
  current_port="${current_port:-8080}"

  if "${venv_dir}/bin/python" - "${current_port}" <<'PY'
import socket
import sys

with socket.socket() as listener:
    try:
        listener.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
  then
    return
  fi

  if docker ps --filter name=lingbot-map-app --format '{{.Ports}}' \
      | grep -Fq "127.0.0.1:${current_port}->8080/tcp"; then
    return
  fi

  selected_port="$("${venv_dir}/bin/python" - <<'PY'
import socket

with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    print(listener.getsockname()[1])
PY
)"
  temporary_env="$(mktemp "${project_dir}/.env.XXXXXX")"
  awk -v port="${selected_port}" '
    BEGIN { replaced=0 }
    /^LINGBOT_PORT=/ && !replaced { print "LINGBOT_PORT=" port; replaced=1; next }
    { print }
    END { if (!replaced) print "LINGBOT_PORT=" port }
  ' "${project_dir}/.env" >"${temporary_env}"
  chmod 600 "${temporary_env}"
  mv "${temporary_env}" "${project_dir}/.env"
  printf 'bootstrap: port %s is occupied; selected loopback port %s\n' \
    "${current_port}" "${selected_port}"
}

ensure_user_linger() {
  local linger_state
  if ! command -v loginctl >/dev/null 2>&1; then
    return
  fi
  linger_state="$(loginctl show-user "$(id -un)" --property=Linger --value 2>/dev/null || true)"
  if [[ "${linger_state}" == "no" ]]; then
    printf 'bootstrap: enabling the supervised user service across logouts\n'
    loginctl enable-linger "$(id -un)" || \
      printf 'warning: could not enable user lingering; the site may stop after logout\n' >&2
  fi
}

bootstrap() {
  ensure_host_tools
  ensure_venv
  ensure_cloudflared
  configure_host_port
  ensure_user_linger
}

wait_until_ready() {
  local timeout deadline url=""
  if [[ -f "${project_dir}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${project_dir}/.env"
    set +a
  fi
  timeout="${LINGBOT_BOOTSTRAP_TIMEOUT:-1800}"
  deadline=$((SECONDS + timeout))
  if [[ "${LINGBOT_DEPLOYMENT_MODE:-public}" == "internal" ]]; then
    printf 'deployment: waiting for the resident model'
  else
    printf 'deployment: waiting for the resident model and public tunnel'
  fi
  while (( SECONDS < deadline )); do
    if ! systemctl --user is-active --quiet lingbot-map.service; then
      if systemctl --user is-failed --quiet lingbot-map.service; then
        printf '\n' >&2
        journalctl --user-unit lingbot-map.service --no-pager -n 80 >&2 || true
        die "lingbot-map.service failed during startup"
      fi
    elif curl -fsS "http://127.0.0.1:${LINGBOT_PORT:-8080}/readyz" >/dev/null 2>&1; then
      url="$(${deploy_script} url 2>/dev/null || true)"
      if [[ -n "${url}" ]]; then
        printf ' ready\n'
        printf '\nLingBot Maproom is ready.\n'
        if [[ "${LINGBOT_DEPLOYMENT_MODE:-public}" == "internal" ]]; then
          printf '  Internal API: %s\n' "${url}"
        else
          printf '  Site:  %s\n' "${url}"
        fi
        printf '  Token: %s\n' "$(${deploy_script} token)"
        printf '  Local: http://127.0.0.1:%s\n' "${LINGBOT_PORT:-8080}"
        printf '\nDrop in one video, click “Build 3D map,” and the result opens in the site.\n'
        return 0
      fi
    fi
    printf '.'
    sleep 5
  done
  printf '\n' >&2
  journalctl --user-unit lingbot-map.service --no-pager -n 80 >&2 || true
  die "deployment did not become ready within ${timeout} seconds"
}

start_supervised() {
  bootstrap
  "${deploy_script}" install-service
  wait_until_ready
}

run_tests() {
  ensure_host_tools
  ensure_venv
  "${venv_dir}/bin/python" -m pip install --disable-pip-version-check pytest Pillow
  (
    cd "${project_dir}"
    "${venv_dir}/bin/python" -m pytest -q
    docker build --target web-builder -t lingbot-map-web-builder:local .
    docker build --target runtime -t lingbot-map-web:local .
  )
}

command_name="${1:-start}"
case "${command_name}" in
  start) start_supervised ;;
  foreground)
    bootstrap
    exec "${deploy_script}" run
    ;;
  restart)
    start_supervised
    ;;
  stop)
    systemctl --user stop lingbot-map.service 2>/dev/null || true
    "${deploy_script}" down
    ;;
  status|logs|url|token|verify)
    exec "${deploy_script}" "${command_name}"
    ;;
  smoke)
    ensure_host_tools
    ensure_venv
    exec "${project_dir}/scripts/smoke-test.sh"
    ;;
  test) run_tests ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
