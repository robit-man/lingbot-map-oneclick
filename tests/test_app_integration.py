from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("multipart", reason="python-multipart runtime dependency is absent")

from fastapi.testclient import TestClient
from PIL import Image

from webapp.app import create_app
from webapp.config import Settings
from webapp.noclip_contract import NOCLIP_REQUEST_SCHEMA


TOKEN = "test-token-that-is-at-least-24-characters"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _settings(tmp_path: Path, monkeypatch) -> Settings:
    monkeypatch.setenv("LINGBOT_API_TOKEN", TOKEN)
    monkeypatch.setenv("LINGBOT_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("LINGBOT_WEB_DIST", str(tmp_path / "web"))
    monkeypatch.setenv("LINGBOT_REQUIRE_GPU_BROKER", "0")
    monkeypatch.setenv("LINGBOT_POINT_CLOUD_MAX_POINTS", "1000")
    return Settings.from_env()


def _png(color: tuple[int, int, int]) -> bytes:
    payload = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(payload, format="PNG")
    return payload.getvalue()


def _manifest(settings: Settings) -> dict:
    return {
        "schema": NOCLIP_REQUEST_SCHEMA,
        "captureSessionId": "capture-integration-1",
        "model": settings.model_repo,
        "revision": settings.model_revision,
        "coordinateSystem": {
            "geodetic": "WGS84",
            "localFrame": "ENU",
            "cameraAxes": "opencv_x_right_y_down_z_forward",
            "quaternionOrder": "xyzw",
            "units": "meters",
        },
        "media": [
            {"sequenceNumber": 0, "monotonicMs": 100.0},
            {"sequenceNumber": 1, "monotonicMs": 200.0},
        ],
        "poses": [
            {"sampleIndex": 0, "monotonicMs": 100.0},
            {"sampleIndex": 1, "monotonicMs": 200.0},
        ],
    }


class ArtifactEngine:
    ready = True
    load_summary = {"device": "test", "weight": "fixture.pt"}

    def load(self):
        self.ready = True
        return self.load_summary

    def reconstruct(
        self,
        *,
        result_dir: Path,
        progress_callback=None,
        cancellation_checkpoint=None,
        noclip_manifest=None,
        **_parameters,
    ):
        if progress_callback:
            progress_callback(70, "Fixture inference", "inference")
        if cancellation_checkpoint:
            cancellation_checkpoint()
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "reconstruction.glb").write_bytes(b"glTF-fixture")
        (result_dir / "point-cloud-lod.ply").write_bytes(b"ply\nfixture")
        trajectory = {
            "schema": "noclip.lingbot.trajectory/1.0",
            "frames": [
                {
                    "sequenceNumber": 0,
                    "sensorSampleIndex": 0,
                    "cameraToWorld": {
                        "positionM": [0, 0, 0],
                        "quaternionXyzw": [0, 0, 0, 1],
                    },
                    "intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0},
                }
            ],
        }
        summary = {"frames_used": 2, "inference_seconds": 0.01}
        files = {
            "trajectory.json": trajectory,
            "intrinsics.json": {"schema": "noclip.lingbot.intrinsics/1.0"},
            "quality-diagnostics.json": {"schema": "noclip.lingbot.quality/1.0"},
            "reconstruction-manifest.json": {
                "schema": "noclip.lingbot.reconstruction/1.0",
                "coordinateContract": {"modelFrame": "exported_lingbot_model"},
                "artifacts": [],
            },
            "summary.json": summary,
        }
        for name, payload in files.items():
            (result_dir / name).write_text(json.dumps(payload), encoding="utf-8")
        return summary


def _submit(
    client: TestClient,
    settings: Settings,
    *,
    key: str = "capture-job-1",
    manifest: dict | None = None,
):
    return client.post(
        "/v1/reconstructions",
        headers={**AUTH, "Idempotency-Key": key},
        data={"manifest": json.dumps(manifest or _manifest(settings))},
        files=[
            ("media", ("000.png", _png((255, 0, 0)), "image/png")),
            ("media", ("001.png", _png((0, 255, 0)), "image/png")),
        ],
    )


def _wait_for_status(client: TestClient, job_id: str, statuses: set[str]):
    deadline = time.monotonic() + 3
    response = None
    while time.monotonic() < deadline:
        response = client.get(f"/v1/reconstructions/{job_id}", headers=AUTH)
        if response.json().get("status") in statuses:
            return response
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {statuses}: {response.json() if response else None}")


def test_only_health_readiness_are_public_operational_endpoints(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    with TestClient(create_app(settings, engine_override=ArtifactEngine())) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        private_requests = [
            ("get", "/api/info"),
            ("post", "/api/jobs"),
            ("get", "/api/jobs/unknown"),
            ("get", "/api/jobs/unknown/result"),
            ("get", "/api/jobs/unknown/summary"),
            ("post", "/v1/reconstructions"),
            ("get", "/v1/reconstructions/unknown"),
            ("delete", "/v1/reconstructions/unknown"),
            ("get", "/v1/reconstructions/unknown/result"),
            ("get", "/v1/reconstructions/unknown/artifacts/summary.json"),
        ]
        for method, path in private_requests:
            assert client.request(method, path).status_code == 401, (method, path)
        assert client.get("/api/info", headers={"X-API-Token": TOKEN}).status_code == 401


def test_authenticated_contract_is_idempotent_and_exposes_capacity_and_artifacts(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path, monkeypatch)
    with TestClient(create_app(settings, engine_override=ArtifactEngine())) as client:
        submitted = _submit(client, settings)
        assert submitted.status_code == 202, submitted.text
        job_id = submitted.json()["jobId"]
        completed = _wait_for_status(client, job_id, {"completed"})
        status = completed.json()
        assert status["stage"] == "ready"
        assert status["activeCapacity"] == 1
        assert status["queueSize"] == 0

        replay = _submit(client, settings)
        assert replay.status_code == 202
        assert replay.json()["jobId"] == job_id
        assert replay.json()["idempotentReplay"] is True

        conflicting_manifest = _manifest(settings)
        conflicting_manifest["captureSessionId"] = "capture-conflict"
        conflict = _submit(client, settings, manifest=conflicting_manifest)
        assert conflict.status_code == 400
        assert "conflicting request" in conflict.json()["detail"]

        result = client.get(f"/v1/reconstructions/{job_id}/result", headers=AUTH)
        assert result.status_code == 200, result.text
        artifacts = result.json()["artifacts"]
        assert {artifact["kind"] for artifact in artifacts} == {
            "reconstruction_glb",
            "point_cloud",
            "trajectory",
            "manifest",
            "confidence",
        }
        for artifact in artifacts:
            assert client.get(artifact["url"], headers=AUTH).status_code == 200
            assert client.get(artifact["url"]).status_code == 401


def test_completed_pre_upgrade_job_remains_readable(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    job_id = "legacy-complete"
    job_dir = settings.jobs_dir / job_id
    result_dir = job_dir / "result"
    result_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "id": job_id,
                "status": "complete",
                "message": "persisted",
                "progress": 100,
                "created_at": "2026-09-01T00:00:00+00:00",
                "updated_at": "2026-09-01T00:00:00+00:00",
                "contract": NOCLIP_REQUEST_SCHEMA,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "reconstruction.glb").write_bytes(b"glTF")
    (result_dir / "trajectory.json").write_text(
        json.dumps({"frames": []}), encoding="utf-8"
    )
    (result_dir / "summary.json").write_text(
        json.dumps({"frames_used": 2}), encoding="utf-8"
    )
    with TestClient(create_app(settings, engine_override=ArtifactEngine())) as client:
        result = client.get(f"/v1/reconstructions/{job_id}/result", headers=AUTH)
        assert result.status_code == 200
        artifacts = result.json()["artifacts"]
        assert {artifact["kind"] for artifact in artifacts} == {
            "reconstruction_glb",
            "trajectory",
            "manifest",
        }
        manifest = next(item for item in artifacts if item["kind"] == "manifest")
        assert manifest["metadata"] == {"legacy": True}


class BlockingEngine(ArtifactEngine):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def reconstruct(self, *, result_dir: Path, progress_callback=None, **_parameters):
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "reconstruction.glb").write_bytes(b"partial-must-not-promote")
        if progress_callback:
            progress_callback(
                55,
                "Fixture non-interruptible model call",
                "inference",
            )
        self.started.set()
        self.release.wait(timeout=3)
        return {"frames_used": 2}


def test_running_cancellation_stays_cancelling_until_model_call_exits(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path, monkeypatch)
    engine = BlockingEngine()
    try:
        with TestClient(create_app(settings, engine_override=engine)) as client:
            submitted = _submit(client, settings, key="capture-cancel-1")
            assert submitted.status_code == 202, submitted.text
            job_id = submitted.json()["jobId"]
            assert engine.started.wait(timeout=2)

            cancelling = client.delete(
                f"/v1/reconstructions/{job_id}", headers=AUTH
            )
            assert cancelling.status_code == 202
            assert cancelling.json()["status"] == "cancelling"
            assert cancelling.json()["cancellationAcknowledgedAt"] is None
            status = client.get(f"/v1/reconstructions/{job_id}", headers=AUTH).json()
            assert status["status"] == "cancelling"

            engine.release.set()
            cancelled = _wait_for_status(client, job_id, {"cancelled"}).json()
            assert cancelled["cancellationAcknowledgedAt"] is not None
            assert not (settings.jobs_dir / job_id / "result").exists()
            assert (
                client.get(f"/v1/reconstructions/{job_id}/result", headers=AUTH).status_code
                == 409
            )
    finally:
        engine.release.set()
