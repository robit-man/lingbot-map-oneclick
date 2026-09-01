from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download

from .config import Settings


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class WeightSpec:
    model_dir: Path
    repo_id: str
    filename: str
    revision: str
    expected_sha256: str = ""
    expected_size: int = 0
    explicit_path: str = ""

    @classmethod
    def from_settings(cls, settings: Settings) -> "WeightSpec":
        return cls(
            model_dir=settings.model_dir,
            repo_id=settings.model_repo,
            filename=settings.model_filename,
            revision=settings.model_revision,
            expected_sha256=settings.model_sha256,
            expected_size=settings.model_size_bytes,
            explicit_path=settings.model_path,
        )


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stamp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.verified.json")


def _read_stamp(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def _write_stamp(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify(path: Path, spec: WeightSpec) -> None:
    if not path.is_file():
        raise RuntimeError(f"checkpoint download did not produce a file: {path}")
    stat = path.stat()
    if spec.expected_size and stat.st_size != spec.expected_size:
        raise RuntimeError(
            f"checkpoint size mismatch for {path}: expected {spec.expected_size}, "
            f"got {stat.st_size}"
        )

    expected_sha = spec.expected_sha256.strip().lower()
    if not expected_sha:
        return
    if not _SHA256_RE.fullmatch(expected_sha):
        raise ValueError("LINGBOT_MODEL_SHA256 must be a lowercase 64-character SHA-256")

    stamp_path = _stamp_path(path)
    stamp = _read_stamp(stamp_path)
    stamp_matches = (
        stamp.get("sha256") == expected_sha
        and stamp.get("size") == stat.st_size
        and stamp.get("mtime_ns") == stat.st_mtime_ns
    )
    if stamp_matches:
        return

    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch for {path}: expected {expected_sha}, "
            f"got {actual_sha}"
        )
    _write_stamp(
        stamp_path,
        {"sha256": actual_sha, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
    )


def pull_weights(
    spec: WeightSpec,
    *,
    downloader: Callable[..., str] = hf_hub_download,
) -> Path:
    """Resolve, download, and verify the configured checkpoint idempotently."""
    spec.model_dir.mkdir(parents=True, exist_ok=True)
    if spec.explicit_path:
        path = Path(spec.explicit_path).expanduser().resolve()
    else:
        existing = (spec.model_dir / spec.filename).resolve()
        if existing.is_file():
            _verify(existing, spec)
            return existing
        path = Path(
            downloader(
                repo_id=spec.repo_id,
                filename=spec.filename,
                revision=spec.revision,
                local_dir=str(spec.model_dir),
            )
        ).resolve()
    _verify(path, spec)
    return path


def main() -> int:
    settings = Settings.from_env(require_token=False)
    spec = WeightSpec.from_settings(settings)
    path = pull_weights(spec)
    print(
        json.dumps(
            {
                "ok": True,
                "path": str(path),
                "repo": spec.repo_id,
                "filename": spec.filename,
                "revision": spec.revision,
                "size": path.stat().st_size,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
