from __future__ import annotations

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
