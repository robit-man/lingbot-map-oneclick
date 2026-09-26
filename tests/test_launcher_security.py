from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_startup_readiness_output_never_expands_the_access_token():
    launcher = (PROJECT_ROOT / "run.sh").read_text()
    wait_body = launcher.split("wait_until_ready() {", 1)[1].split(
        "start_supervised() {", 1
    )[0]

    assert '${deploy_script} token' not in wait_body
    assert "bearer token configured" in wait_body


def test_token_rotation_is_silent_and_replaces_the_credential(tmp_path):
    shutil.copy(PROJECT_ROOT / ".env.example", tmp_path / ".env.example")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    deploy = scripts / "deploy.sh"
    shutil.copy(PROJECT_ROOT / "scripts" / "deploy.sh", deploy)

    first = subprocess.run(
        [deploy, "token"], check=True, capture_output=True, text=True
    ).stdout.strip()
    rotated = subprocess.run(
        [deploy, "rotate-token"], check=True, capture_output=True, text=True
    )
    second = subprocess.run(
        [deploy, "token"], check=True, capture_output=True, text=True
    ).stdout.strip()

    assert rotated.stdout == ""
    assert rotated.stderr == ""
    assert len(first) == 48
    assert len(second) == 48
    assert second != first
