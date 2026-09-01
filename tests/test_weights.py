from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from webapp import weights
from webapp.weights import WeightSpec, pull_weights


def _spec(tmp_path: Path, payload: bytes) -> WeightSpec:
    return WeightSpec(
        model_dir=tmp_path,
        repo_id="owner/model",
        filename="model.pt",
        revision="deadbeef",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
    )


def test_pull_weights_downloads_and_reuses_integrity_stamp(tmp_path, monkeypatch):
    payload = b"test-checkpoint"
    checkpoint = tmp_path / "model.pt"
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        checkpoint.write_bytes(payload)
        return str(checkpoint)

    spec = _spec(tmp_path, payload)
    assert pull_weights(spec, downloader=downloader) == checkpoint
    assert calls == [
        {
            "repo_id": "owner/model",
            "filename": "model.pt",
            "revision": "deadbeef",
            "local_dir": str(tmp_path),
        }
    ]

    def unexpected_hash(_path):
        raise AssertionError("verified file should use its integrity stamp")

    monkeypatch.setattr(weights, "sha256_file", unexpected_hash)
    assert pull_weights(
        spec,
        downloader=lambda **_kwargs: pytest.fail("verified file should be reused"),
    ) == checkpoint


def test_pull_weights_rejects_bad_checksum(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"wrong")
    spec = WeightSpec(
        model_dir=tmp_path,
        repo_id="owner/model",
        filename="model.pt",
        revision="deadbeef",
        expected_sha256="0" * 64,
        expected_size=5,
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        pull_weights(spec, downloader=lambda **_kwargs: str(checkpoint))


def test_explicit_path_skips_downloader(tmp_path):
    checkpoint = tmp_path / "custom.pt"
    checkpoint.write_bytes(b"custom")
    spec = WeightSpec(
        model_dir=tmp_path,
        repo_id="unused",
        filename="unused.pt",
        revision="unused",
        explicit_path=str(checkpoint),
    )
    assert pull_weights(
        spec,
        downloader=lambda **_kwargs: pytest.fail("downloader should not run"),
    ) == checkpoint
