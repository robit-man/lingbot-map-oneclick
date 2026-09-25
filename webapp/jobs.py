from __future__ import annotations

import json
import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .engine import InferenceEngine


LOGGER = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ReconstructionCancelled(RuntimeError):
    """Raised at a cooperative checkpoint after cancellation is requested."""


@dataclass
class Job:
    id: str
    status: str = "queued"
    stage: str = "provider_queued"
    message: str = "Waiting for provider capacity"
    progress: int = 5
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    summary: dict[str, Any] | None = None
    contract: str | None = None
    request_fingerprint: str | None = None
    cancellation_requested_at: str | None = None
    cancellation_acknowledged_at: str | None = None

    def public(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("request_fingerprint", None)
        if self.status == "complete":
            payload["result_url"] = f"/api/jobs/{self.id}/result"
            payload["summary_url"] = f"/api/jobs/{self.id}/summary"
        return payload


class QueueFullError(RuntimeError):
    pass


InputPreprocessor = Callable[..., dict[str, Any]]


class JobManager:
    ACTIVE_STATUSES = {"staging", "queued", "running", "cancelling"}
    TERMINAL_STATUSES = {"complete", "failed", "cancelled"}

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
        self._cancellations: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lingbot")
        self._load_retained()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancellation_checkpoint(self, job_id: str) -> None:
        with self._lock:
            cancellation = self._cancellations.get(job_id)
            cancelled = cancellation is not None and cancellation.is_set()
        if cancelled:
            raise ReconstructionCancelled(job_id)

    def public(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            payload = job.public()
            queued = sorted(
                (
                    item
                    for item in self._jobs.values()
                    if item.status in {"staging", "queued"}
                ),
                key=lambda item: (item.created_at, item.id),
            )
            active = sum(
                item.status in {"running", "cancelling"}
                for item in self._jobs.values()
            )
            payload.update(
                {
                    "queuePosition": queued.index(job) + 1 if job in queued else None,
                    "queueSize": len(queued),
                    "activeJobs": active,
                    "activeCapacity": 1,
                    "availableCapacity": max(0, 1 - active),
                    "maxQueue": self.max_queue,
                    "estimatedWaitMinSeconds": None,
                    "estimatedWaitMaxSeconds": None,
                }
            )
            return payload

    def capacity(self) -> dict[str, int]:
        with self._lock:
            queued = sum(
                item.status in {"staging", "queued"}
                for item in self._jobs.values()
            )
            active = sum(
                item.status in {"running", "cancelling"}
                for item in self._jobs.values()
            )
            return {
                "queueSize": queued,
                "activeJobs": active,
                "activeCapacity": 1,
                "availableCapacity": max(0, 1 - active),
                "maxQueue": self.max_queue,
            }

    def reserve(
        self,
        job_id: str,
        *,
        contract: str | None = None,
        request_fingerprint: str | None = None,
    ) -> tuple[Job, bool]:
        """Atomically reserve an idempotent job ID before request staging begins."""

        with self._lock:
            existing = self._jobs.get(job_id)
            if existing is not None:
                if existing.contract != contract:
                    raise ValueError("job id is already reserved for another contract")
                if (
                    request_fingerprint is not None
                    and existing.request_fingerprint is not None
                    and existing.request_fingerprint != request_fingerprint
                ):
                    raise ValueError(
                        "idempotency key was reused with a conflicting request"
                    )
                return existing, False
            active = sum(
                job.status in self.ACTIVE_STATUSES for job in self._jobs.values()
            )
            if active >= self.max_queue:
                raise QueueFullError("the reconstruction queue is full")
            job = Job(
                id=job_id,
                status="staging",
                stage="staging",
                message="Receiving bounded provider input",
                progress=2,
                contract=contract,
                request_fingerprint=request_fingerprint,
            )
            self._jobs[job_id] = job
            self._cancellations[job_id] = threading.Event()
            self._persist(job)
            return job, True

    def enqueue(
        self,
        job_id: str,
        *,
        preprocessor: InputPreprocessor | None = None,
        **parameters: Any,
    ) -> Job:
        cancelled_during_staging = False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError("job reservation does not exist")
            if job.status == "cancelling":
                job.status = "cancelled"
                job.stage = "cancelled"
                job.message = "Cancellation acknowledged after upload staging"
                job.cancellation_acknowledged_at = _now()
                job.updated_at = _now()
                self._persist(job)
                cancelled_during_staging = True
            elif job.status != "staging":
                raise ValueError(f"job cannot be enqueued from {job.status}")
            else:
                job.status = "queued"
                job.stage = "provider_queued"
                job.message = "Waiting for provider capacity"
                job.progress = 5
                job.updated_at = _now()
                self._persist(job)
        if cancelled_during_staging:
            self._cleanup_temporaries(job_id, remove_result=True)
            return job
        self._executor.submit(self._execute, job, preprocessor, parameters)
        return job

    def submit(
        self,
        job_id: str,
        *,
        contract: str | None = None,
        request_fingerprint: str | None = None,
        preprocessor: InputPreprocessor | None = None,
        **parameters: Any,
    ) -> Job:
        job, created = self.reserve(
            job_id,
            contract=contract,
            request_fingerprint=request_fingerprint,
        )
        if not created:
            return job
        return self.enqueue(job_id, preprocessor=preprocessor, **parameters)

    def fail_reserved(self, job_id: str, message: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            cancellation = self._cancellations.get(job_id)
            if cancellation is not None and cancellation.is_set():
                job.status = "cancelled"
                job.stage = "cancelled"
                job.message = "Cancellation acknowledged during staging"
                job.cancellation_acknowledged_at = _now()
            elif job.status == "staging":
                job.status = "failed"
                job.stage = "failed"
                job.message = message
            job.updated_at = _now()
            self._persist(job)
        self._cleanup_temporaries(job_id, remove_result=job.status == "cancelled")
        return job

    def cancel(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in self.ACTIVE_STATUSES:
                return job
            cancellation = self._cancellations.setdefault(job_id, threading.Event())
            cancellation.set()
            job.cancellation_requested_at = job.cancellation_requested_at or _now()
            if job.status == "queued":
                job.status = "cancelled"
                job.stage = "cancelled"
                job.message = "Cancelled before provider execution"
                job.cancellation_acknowledged_at = _now()
            else:
                current_stage = job.stage
                job.status = "cancelling"
                job.stage = f"cancelling_{current_stage}"
                job.message = (
                    "Cancellation requested; waiting for the current safe checkpoint"
                )
            job.updated_at = _now()
            self._persist(job)
        if job.status == "cancelled":
            self._cleanup_temporaries(job_id, remove_result=True)
        return job

    def _load_retained(self) -> None:
        """Restore durable results and make interrupted work terminal after restart."""

        allowed = {field.name for field in Job.__dataclass_fields__.values()}
        for job_file in sorted(self.root.glob("*/job.json")):
            try:
                payload = json.loads(job_file.read_text(encoding="utf-8"))
                job = Job(
                    **{key: value for key, value in payload.items() if key in allowed}
                )
                if "stage" not in payload and job.status in self.TERMINAL_STATUSES:
                    job.stage = "ready" if job.status == "complete" else job.status
                if job.id != job_file.parent.name:
                    raise ValueError("persisted job id does not match its directory")
                if job.status == "cancelling":
                    job.status = "cancelled"
                    job.stage = "cancelled"
                    job.message = "Cancellation acknowledged by provider restart"
                    job.cancellation_acknowledged_at = _now()
                    job.updated_at = _now()
                    self._persist(job)
                    self._cleanup_temporaries(job.id, remove_result=True)
                elif job.status in {"staging", "queued", "running"}:
                    job.status = "failed"
                    job.stage = "failed_restart_interrupted"
                    job.message = "Reconstruction was interrupted by a service restart"
                    job.updated_at = _now()
                    self._persist(job)
                    self._cleanup_temporaries(job.id)
                self._jobs[job.id] = job
                self._cancellations[job.id] = threading.Event()
            except Exception:
                LOGGER.warning(
                    "ignoring invalid persisted job %s", job_file, exc_info=True
                )
        self._prune()

    def _persist(self, job: Job) -> None:
        job_dir = self.root / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        temporary = job_dir / ".job.json.tmp"
        temporary.write_text(
            json.dumps(asdict(job), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(job_dir / "job.json")

    def _update(self, job: Job, **changes: Any) -> bool:
        with self._lock:
            if job.status == "cancelled":
                return False
            if job.status == "cancelling" and changes.get("status") != "cancelled":
                return False
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _now()
            self._persist(job)
            return True

    def _checkpoint(self, job: Job) -> None:
        cancellation = self._cancellations.get(job.id)
        if cancellation is not None and cancellation.is_set():
            raise ReconstructionCancelled(job.id)

    def _acknowledge_cancellation(self, job: Job, message: str) -> None:
        # Remove every promotable artifact before making the terminal status
        # observable. Callers must never see cancelled while output still exists.
        self._cleanup_temporaries(job.id, remove_result=True)
        with self._lock:
            job.status = "cancelled"
            job.stage = "cancelled"
            job.message = message
            job.cancellation_requested_at = job.cancellation_requested_at or _now()
            job.cancellation_acknowledged_at = _now()
            job.updated_at = _now()
            self._persist(job)

    def _execute(
        self,
        job: Job,
        preprocessor: InputPreprocessor | None,
        parameters: dict[str, Any],
    ) -> None:
        try:
            self._checkpoint(job)
        except ReconstructionCancelled:
            return
        self._update(
            job,
            status="running",
            stage="preprocessing",
            message="Preparing bounded frames for reconstruction",
            progress=10,
        )

        def report_progress(
            percent: int,
            message: str,
            stage: str = "reconstructing",
        ) -> None:
            self._checkpoint(job)
            self._update(
                job,
                status="running",
                stage=stage,
                message=message,
                progress=max(10, min(99, int(percent))),
            )

        try:
            if preprocessor is not None:
                prepared = preprocessor(
                    cancellation_checkpoint=lambda: self._checkpoint(job),
                    progress_callback=report_progress,
                )
                parameters.update(prepared)
            self._checkpoint(job)
            summary = self.engine.reconstruct(
                **parameters,
                progress_callback=report_progress,
                cancellation_checkpoint=lambda: self._checkpoint(job),
            )
            self._checkpoint(job)
        except ReconstructionCancelled:
            self._acknowledge_cancellation(
                job, "Cancellation acknowledged after provider cleanup"
            )
            return
        except Exception as exc:
            cancellation = self._cancellations.get(job.id)
            if cancellation is not None and cancellation.is_set():
                self._acknowledge_cancellation(
                    job, "Cancellation acknowledged after provider cleanup"
                )
                return
            LOGGER.exception("job %s failed", job.id)
            self._update(
                job,
                status="failed",
                stage="failed",
                message=f"Reconstruction failed: {type(exc).__name__}: {exc}",
            )
            self._cleanup_temporaries(job.id)
            return
        completed = self._update(
            job,
            status="complete",
            stage="ready",
            message="Reconstruction complete",
            progress=100,
            summary=summary,
        )
        if not completed:
            cancellation = self._cancellations.get(job.id)
            if cancellation is not None and cancellation.is_set():
                self._acknowledge_cancellation(
                    job, "Cancellation acknowledged before result promotion"
                )
            return
        self._cleanup_temporaries(job.id)
        self._prune()

    def _cleanup_temporaries(self, job_id: str, *, remove_result: bool = False) -> None:
        job_dir = self.root / job_id
        shutil.rmtree(job_dir / "upload", ignore_errors=True)
        shutil.rmtree(job_dir / "frames", ignore_errors=True)
        if remove_result:
            shutil.rmtree(job_dir / "result", ignore_errors=True)

    def _prune(self) -> None:
        with self._lock:
            finished = sorted(
                (
                    job
                    for job in self._jobs.values()
                    if job.status in self.TERMINAL_STATUSES
                ),
                key=lambda item: item.updated_at,
            )
            stale = finished[: max(0, len(finished) - self.retain_jobs)]
            for job in stale:
                self._jobs.pop(job.id, None)
                self._cancellations.pop(job.id, None)
                shutil.rmtree(self.root / job.id, ignore_errors=True)
