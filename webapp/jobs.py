from __future__ import annotations

import json
import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .engine import InferenceEngine


LOGGER = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Job:
    id: str
    status: str = "queued"
    message: str = "Waiting for the GPU"
    progress: int = 25
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    summary: dict[str, Any] | None = None

    def public(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.status == "complete":
            payload["result_url"] = f"/api/jobs/{self.id}/result"
            payload["summary_url"] = f"/api/jobs/{self.id}/summary"
        return payload


class QueueFullError(RuntimeError):
    pass


class JobManager:
    def __init__(
        self,
        *,
        engine: InferenceEngine,
        root: Path,
        max_queue: int,
        retain_jobs: int,
    ):
        self.engine = engine
        self.root = root
        self.max_queue = max_queue
        self.retain_jobs = retain_jobs
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lingbot")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _persist(self, job: Job) -> None:
        job_dir = self.root / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        temporary = job_dir / ".job.json.tmp"
        temporary.write_text(
            json.dumps(job.public(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(job_dir / "job.json")

    def _update(self, job: Job, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _now()
            self._persist(job)

    def submit(self, job_id: str, **parameters: Any) -> Job:
        with self._lock:
            active = sum(
                job.status in {"queued", "running"} for job in self._jobs.values()
            )
            if active >= self.max_queue:
                raise QueueFullError("the reconstruction queue is full")
            job = Job(id=job_id)
            self._jobs[job_id] = job
            self._persist(job)
        self._executor.submit(self._execute, job, parameters)
        return job

    def _execute(self, job: Job, parameters: dict[str, Any]) -> None:
        self._update(
            job,
            status="running",
            message="Preparing frames for reconstruction",
            progress=30,
        )

        def report_progress(percent: int, message: str) -> None:
            self._update(
                job,
                status="running",
                message=message,
                progress=max(30, min(99, int(percent))),
            )

        try:
            summary = self.engine.reconstruct(
                **parameters,
                progress_callback=report_progress,
            )
        except Exception as exc:
            LOGGER.exception("job %s failed", job.id)
            self._update(
                job,
                status="failed",
                message=f"Reconstruction failed: {type(exc).__name__}: {exc}",
            )
            return
        self._update(
            job,
            status="complete",
            message="Reconstruction complete",
            progress=100,
            summary=summary,
        )
        self._prune()

    def _prune(self) -> None:
        with self._lock:
            finished = sorted(
                (
                    job
                    for job in self._jobs.values()
                    if job.status in {"complete", "failed"}
                ),
                key=lambda item: item.updated_at,
            )
            stale = finished[: max(0, len(finished) - self.retain_jobs)]
            for job in stale:
                self._jobs.pop(job.id, None)
                shutil.rmtree(self.root / job.id, ignore_errors=True)
