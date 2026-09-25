from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from webapp.jobs import JobManager


class FakeEngine:
    def reconstruct(self, *, result_dir: Path, progress_callback=None, **_parameters):
        if progress_callback:
            progress_callback(80, "Building the map")
        result_dir.mkdir(parents=True)
        (result_dir / "reconstruction.glb").write_bytes(b"glTF")
        return {"frames_used": 2}


def test_job_manager_runs_and_persists_job(tmp_path: Path):
    manager = JobManager(
        engine=FakeEngine(), root=tmp_path, max_queue=2, retain_jobs=2
    )
    try:
        job = manager.submit(
            "job-1",
            result_dir=tmp_path / "job-1" / "result",
            image_dir=tmp_path / "job-1" / "frames",
        )
        deadline = time.monotonic() + 2
        while job.status not in {"complete", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.status == "complete"
        assert job.progress == 100
        assert job.public()["result_url"] == "/api/jobs/job-1/result"
        assert (tmp_path / "job-1" / "job.json").is_file()
    finally:
        manager.close()


def test_job_manager_restores_terminal_jobs_and_resolves_interrupted_jobs(tmp_path: Path):
    for job_id, status in (
        ("finished", "complete"),
        ("interrupted", "running"),
        ("cancellation", "cancelling"),
    ):
        job_dir = tmp_path / job_id
        job_dir.mkdir()
        (job_dir / "job.json").write_text(json.dumps({
            "id": job_id,
            "status": status,
            "message": "persisted",
            "progress": 100 if status == "complete" else 60,
            "created_at": "2026-09-01T00:00:00+00:00",
            "updated_at": "2026-09-01T00:00:00+00:00",
            "summary": {"frames_used": 2} if status == "complete" else None,
            "contract": "noclip.lingbot.request/1.0",
            "result_url": "/ignored-derived-field",
        }), encoding="utf-8")

    manager = JobManager(
        engine=FakeEngine(), root=tmp_path, max_queue=2, retain_jobs=4
    )
    try:
        assert manager.get("finished").status == "complete"
        interrupted = manager.get("interrupted")
        assert interrupted.status == "failed"
        assert "service restart" in interrupted.message
        cancellation = manager.get("cancellation")
        assert cancellation.status == "cancelled"
        assert cancellation.cancellation_acknowledged_at is not None
    finally:
        manager.close()


class BlockingEngine:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def reconstruct(self, *, result_dir: Path, progress_callback=None, **_parameters):
        self.started.set()
        self.release.wait(timeout=2)
        return {"frames_used": 2}


def test_running_job_remains_cancelling_until_gpu_work_returns(tmp_path: Path):
    engine = BlockingEngine()
    manager = JobManager(engine=engine, root=tmp_path, max_queue=2, retain_jobs=2)
    try:
        job = manager.submit("cancel-running", result_dir=tmp_path / "cancel-running" / "result")
        assert engine.started.wait(timeout=1)
        manager.cancel(job.id)
        assert job.status == "cancelling"
        assert job.cancellation_requested_at is not None
        assert job.cancellation_acknowledged_at is None
        engine.release.set()
        deadline = time.monotonic() + 1
        while job.status != "cancelled" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.status == "cancelled"
        assert job.cancellation_acknowledged_at is not None
        persisted = json.loads((tmp_path / job.id / "job.json").read_text(encoding="utf-8"))
        assert persisted["status"] == "cancelled"
    finally:
        engine.release.set()
        manager.close()


class FinalizationRaceManager(JobManager):
    """Inject a cancellation at the final checkpoint/update boundary."""

    def _update(self, job, **changes):
        if changes.get("status") == "complete":
            self.cancel(job.id)
        return super()._update(job, **changes)


def test_cancellation_racing_result_promotion_reaches_terminal_state(tmp_path: Path):
    manager = FinalizationRaceManager(
        engine=FakeEngine(), root=tmp_path, max_queue=2, retain_jobs=2
    )
    try:
        job = manager.submit(
            "cancel-finalize",
            result_dir=tmp_path / "cancel-finalize" / "result",
        )
        deadline = time.monotonic() + 2
        while job.status not in {"cancelled", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.status == "cancelled"
        assert job.cancellation_acknowledged_at is not None
        assert not (tmp_path / job.id / "result").exists()
    finally:
        manager.close()
