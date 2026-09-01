from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GpuLeaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuLease:
    token: str
    socket_path: Path
    required: bool = True
    timeout_seconds: float = 300.0

    @classmethod
    def from_env(cls, *, required: bool = True) -> "GpuLease":
        return cls(
            token=os.getenv("OLLAMA_UNIFY_GPU_LEASE", "").strip(),
            socket_path=Path(
                os.getenv(
                    "OLLAMA_UNIFY_SOCKET",
                    "/run/ollama-unify/gpu-negotiator.sock",
                )
            ),
            required=required,
        )

    @property
    def configured(self) -> bool:
        return bool(self.token)

    def _control(self, action: str) -> dict[str, Any]:
        if not self.token:
            if self.required:
                raise GpuLeaseError(
                    "missing OLLAMA_UNIFY_GPU_LEASE; start with scripts/deploy.sh"
                )
            return {"ok": True, "skipped": True}
        if not self.socket_path.is_socket():
            raise GpuLeaseError(
                f"GPU broker control socket is unavailable: {self.socket_path}"
            )

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        response_file = None
        try:
            client.settimeout(self.timeout_seconds)
            client.connect(str(self.socket_path))
            client.sendall(
                json.dumps({"action": action, "token": self.token}).encode("utf-8")
                + b"\n"
            )
            response_file = client.makefile("rb")
            raw = response_file.readline(1024 * 1024)
            if not raw:
                raise GpuLeaseError("GPU broker closed the control connection")
            response = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise GpuLeaseError(f"GPU broker {action} failed: {exc}") from exc
        finally:
            if response_file is not None:
                response_file.close()
            client.close()

        if not response.get("ok"):
            raise GpuLeaseError(
                f"GPU broker {action} failed: {response.get('error', 'unknown error')}"
            )
        return response

    def prepare(self) -> dict[str, Any]:
        """Drain/refit broker-owned lanes before inference grows CUDA memory."""
        return self._control("prepare")

    def ready(self) -> dict[str, Any]:
        """Report the stable post-growth allocation back to the broker."""
        return self._control("ready")
