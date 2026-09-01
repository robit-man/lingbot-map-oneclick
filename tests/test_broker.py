from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest

from webapp.broker import GpuLease, GpuLeaseError


def _serve(socket_path: Path, actions: list[str]) -> threading.Thread:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen()

    def run():
        try:
            for expected_action in actions:
                connection, _ = server.accept()
                with connection:
                    request_file = connection.makefile("rb")
                    request = json.loads(request_file.readline())
                    assert request == {"action": expected_action, "token": "lease-test"}
                    connection.sendall(
                        json.dumps({"ok": True, "action": expected_action}).encode()
                        + b"\n"
                    )
        finally:
            server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_prepare_and_ready_use_scoped_broker_protocol(tmp_path):
    socket_path = tmp_path / "broker.sock"
    thread = _serve(socket_path, ["prepare", "ready"])
    lease = GpuLease(token="lease-test", socket_path=socket_path)
    assert lease.prepare()["action"] == "prepare"
    assert lease.ready()["action"] == "ready"
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_required_lease_rejects_unbrokered_growth(tmp_path):
    lease = GpuLease(token="", socket_path=tmp_path / "missing.sock", required=True)
    with pytest.raises(GpuLeaseError, match="missing OLLAMA_UNIFY_GPU_LEASE"):
        lease.prepare()


def test_optional_lease_is_explicitly_skipped(tmp_path):
    lease = GpuLease(token="", socket_path=tmp_path / "missing.sock", required=False)
    assert lease.prepare() == {"ok": True, "skipped": True}
