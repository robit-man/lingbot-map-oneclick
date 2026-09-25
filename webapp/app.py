from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .engine import InferenceEngine
from .inputs import (
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    stage_input_media,
)
from .jobs import JobManager, QueueFullError, ReconstructionCancelled
from .noclip_contract import (
    NOCLIP_REQUEST_SCHEMA,
    NOCLIP_RESULT_SCHEMA,
    validate_noclip_manifest,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


async def _save_uploads(
    uploads: list[UploadFile],
    destination: Path,
    *,
    byte_limit: int,
    file_limit: int,
    cancellation_checkpoint=None,
) -> list[Path]:
    if not 1 <= len(uploads) <= file_limit:
        raise HTTPException(400, f"upload between 1 and {file_limit} files")
    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    saved: list[Path] = []
    try:
        for index, upload in enumerate(uploads):
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
                raise HTTPException(400, f"unsupported file type: {suffix or 'none'}")
            target = destination / f"{index:06d}{suffix}"
            with target.open("wb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    if cancellation_checkpoint is not None:
                        cancellation_checkpoint()
                    total += len(chunk)
                    if total > byte_limit:
                        raise HTTPException(
                            413,
                            f"upload exceeds the {byte_limit // (1024 * 1024)} MiB limit",
                        )
                    handle.write(chunk)
            saved.append(target)
            await upload.close()
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return saved


def create_app(
    settings: Settings | None = None,
    *,
    engine_override: InferenceEngine | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    engine = engine_override or InferenceEngine(settings)
    manager: JobManager | None = None

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        nonlocal manager
        settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(engine.load)
        manager = JobManager(
            engine=engine,
            root=settings.jobs_dir,
            max_queue=settings.max_queue,
            retain_jobs=settings.retain_jobs,
        )
        application.state.engine = engine
        application.state.jobs = manager
        yield
        manager.close()

    application = FastAPI(
        title="LingBot-Map",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data: blob:; connect-src 'self'; "
            "worker-src 'self' blob:; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'"
        )
        return response

    def require_token(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        bearer = ""
        if authorization and authorization.lower().startswith("bearer "):
            bearer = authorization[7:].strip()
        if not bearer or not hmac.compare_digest(bearer, settings.api_token):
            raise HTTPException(
                401,
                "invalid access token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    auth = Depends(require_token)

    @application.get("/healthz")
    async def healthz():
        return {"ok": True}

    @application.get("/readyz")
    async def readyz():
        if not engine.ready:
            return JSONResponse({"ok": False, "status": "loading"}, status_code=503)
        return {"ok": True, "status": "ready", "model": engine.load_summary}

    @application.get("/api/info", dependencies=[auth])
    async def info():
        return {
            "ready": engine.ready,
            "model": engine.load_summary,
            "limits": {
                "upload_bytes": settings.max_upload_bytes,
                "files": settings.max_files,
                "frames": settings.max_frames,
                "queue": settings.max_queue,
            },
            "capacity": manager.capacity() if manager is not None else None,
        }

    @application.post("/api/jobs", dependencies=[auth], status_code=202)
    async def create_job(
        files: Annotated[list[UploadFile], File()],
        fps: Annotated[int, Form()] = 8,
        max_frames: Annotated[int, Form()] = 48,
        num_scale_frames: Annotated[int, Form()] = 4,
        keyframe_interval: Annotated[int, Form()] = 2,
        confidence_percentile: Annotated[float, Form()] = 50.0,
        include_cameras: Annotated[bool, Form()] = True,
    ):
        if not 1 <= fps <= 30:
            raise HTTPException(400, "fps must be between 1 and 30")
        if not 2 <= max_frames <= settings.max_frames:
            raise HTTPException(
                400, f"max_frames must be between 2 and {settings.max_frames}"
            )
        if not 1 <= num_scale_frames <= 8:
            raise HTTPException(400, "num_scale_frames must be between 1 and 8")
        if not 1 <= keyframe_interval <= 16:
            raise HTTPException(400, "keyframe_interval must be between 1 and 16")
        if not 0 <= confidence_percentile <= 95:
            raise HTTPException(400, "confidence_percentile must be between 0 and 95")
        if manager is None:
            raise HTTPException(503, "job manager is not ready")

        job_id = uuid.uuid4().hex
        job_dir = settings.jobs_dir / job_id
        upload_dir = job_dir / "upload"
        try:
            manager.reserve(job_id, contract="browser")
            uploaded = await _save_uploads(
                files,
                upload_dir,
                byte_limit=settings.max_upload_bytes,
                file_limit=settings.max_files,
                cancellation_checkpoint=lambda: manager.cancellation_checkpoint(job_id),
            )
            image_uploads = [path for path in uploaded if path.suffix in IMAGE_SUFFIXES]
            video_uploads = [path for path in uploaded if path.suffix in VIDEO_SUFFIXES]
            if video_uploads and image_uploads:
                raise HTTPException(400, "upload either one video or an image sequence")
            if len(video_uploads) > 1:
                raise HTTPException(400, "upload only one video")
            if not video_uploads and len(image_uploads) < 2:
                raise HTTPException(400, "upload at least two images")
            job = manager.enqueue(
                job_id,
                preprocessor=partial(
                    stage_input_media,
                    uploaded,
                    job_dir / "frames",
                    fps=fps,
                    max_frames=max_frames,
                ),
                result_dir=job_dir / "result",
                num_scale_frames=num_scale_frames,
                keyframe_interval=keyframe_interval,
                confidence_percentile=confidence_percentile,
                include_cameras=include_cameras,
            )
            return manager.public(job.id)
        except QueueFullError as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(429, str(exc)) from exc
        except ReconstructionCancelled as exc:
            manager.fail_reserved(job_id, "Cancellation acknowledged during upload")
            raise HTTPException(409, "upload was cancelled") from exc
        except HTTPException:
            manager.fail_reserved(job_id, "Browser upload validation failed")
            raise
        except ValueError as exc:
            manager.fail_reserved(job_id, f"Browser upload validation failed: {exc}")
            raise HTTPException(400, str(exc)) from exc
        except Exception:
            manager.fail_reserved(job_id, "Browser upload staging failed")
            LOGGER.exception("failed to stage job %s", job_id)
            raise HTTPException(500, "could not stage the upload")

    @application.get("/api/jobs/{job_id}", dependencies=[auth])
    async def job_status(job_id: str):
        if manager is None:
            raise HTTPException(503, "job manager is not ready")
        payload = manager.public(job_id)
        if payload is None:
            raise HTTPException(404, "job not found")
        return payload

    @application.get("/api/jobs/{job_id}/result", dependencies=[auth])
    async def job_result(job_id: str):
        if manager is None:
            raise HTTPException(503, "job manager is not ready")
        job = manager.get(job_id)
        if job is None or job.status != "complete":
            raise HTTPException(404, "completed result not found")
        path = settings.jobs_dir / job_id / "result" / "reconstruction.glb"
        if not path.is_file():
            raise HTTPException(404, "result file not found")
        return FileResponse(
            path,
            media_type="model/gltf-binary",
            filename=f"lingbot-map-{job_id[:8]}.glb",
        )

    @application.get("/api/jobs/{job_id}/summary", dependencies=[auth])
    async def job_summary(job_id: str):
        if manager is None:
            raise HTTPException(503, "job manager is not ready")
        job = manager.get(job_id)
        if job is None or job.status != "complete":
            raise HTTPException(404, "completed summary not found")
        path = settings.jobs_dir / job_id / "result" / "summary.json"
        if not path.is_file():
            raise HTTPException(404, "summary file not found")
        return FileResponse(path, media_type="application/json")

    @application.post("/v1/reconstructions", dependencies=[auth], status_code=202)
    async def create_noclip_reconstruction(
        media: Annotated[list[UploadFile], File()],
        manifest: Annotated[str, Form()],
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
        model: Annotated[str | None, Form()] = None,
        revision: Annotated[str | None, Form()] = None,
    ):
        try:
            manifest_data = validate_noclip_manifest(json.loads(manifest))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        requested_model = model or manifest_data.get("model") or settings.model_repo
        requested_revision = (
            revision or manifest_data.get("revision") or settings.model_revision
        )
        if requested_model != settings.model_repo:
            raise HTTPException(409, "requested model is not resident on this worker")
        if requested_revision != settings.model_revision:
            raise HTTPException(409, "requested revision is not resident on this worker")
        if not 8 <= len(idempotency_key) <= 200 or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:"
            for character in idempotency_key
        ):
            raise HTTPException(400, "Idempotency-Key is invalid")
        if manager is None:
            raise HTTPException(503, "job manager is not ready")
        job_id = hashlib.sha256(
            f"noclip.lingbot:{idempotency_key}".encode("utf-8")
        ).hexdigest()
        request_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "manifest": manifest_data,
                    "model": requested_model,
                    "revision": requested_revision,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        job_dir = settings.jobs_dir / job_id
        try:
            _job, created = manager.reserve(
                job_id,
                contract=NOCLIP_REQUEST_SCHEMA,
                request_fingerprint=request_fingerprint,
            )
            if not created:
                payload = manager.public(job_id) or {}
                return {
                    "jobId": job_id,
                    "status": payload.get("status", "failed"),
                    "stage": payload.get("stage", "failed"),
                    "queuePosition": payload.get("queuePosition"),
                    "queueSize": payload.get("queueSize", 0),
                    "activeCapacity": payload.get("activeCapacity", 1),
                    "availableCapacity": payload.get("availableCapacity", 0),
                    "idempotentReplay": True,
                }
            uploaded = await _save_uploads(
                media,
                job_dir / "upload",
                byte_limit=settings.max_upload_bytes,
                file_limit=settings.max_files,
                cancellation_checkpoint=lambda: manager.cancellation_checkpoint(job_id),
            )
            image_uploads = [path for path in uploaded if path.suffix in IMAGE_SUFFIXES]
            video_uploads = [path for path in uploaded if path.suffix in VIDEO_SUFFIXES]
            if video_uploads and image_uploads:
                raise HTTPException(400, "upload either one video or an image sequence")
            if len(video_uploads) > 1:
                raise HTTPException(400, "upload only one video")
            options = manifest_data.get("options") or {}
            max_frames = min(
                settings.max_frames,
                max(2, int(options.get("maxFrames", settings.max_frames))),
            )
            fps = min(30, max(1, int(options.get("videoFps", 8))))
            if not video_uploads and len(image_uploads) < 2:
                raise HTTPException(400, "upload at least two images")
            job = manager.enqueue(
                job_id,
                preprocessor=partial(
                    stage_input_media,
                    uploaded,
                    job_dir / "frames",
                    fps=fps,
                    max_frames=max_frames,
                    summary_metadata={
                        "model": settings.model_repo,
                        "revision": settings.model_revision,
                    },
                ),
                result_dir=job_dir / "result",
                num_scale_frames=min(
                    8, max(1, int(options.get("numScaleFrames", settings.default_scale_frames)))
                ),
                keyframe_interval=min(
                    16, max(1, int(options.get("keyframeInterval", 2)))
                ),
                confidence_percentile=min(
                    95.0, max(0.0, float(options.get("confidencePercentile", 50.0)))
                ),
                include_cameras=bool(options.get("includeCameras", False)),
                noclip_manifest=manifest_data,
            )
            payload = manager.public(job.id) or {}
            return {
                "jobId": job.id,
                "status": payload.get("status", "queued"),
                "stage": payload.get("stage", "provider_queued"),
                "queuePosition": payload.get("queuePosition"),
                "queueSize": payload.get("queueSize", 0),
                "activeCapacity": payload.get("activeCapacity", 1),
                "availableCapacity": payload.get("availableCapacity", 0),
                "estimatedWaitMinSeconds": payload.get("estimatedWaitMinSeconds"),
                "estimatedWaitMaxSeconds": payload.get("estimatedWaitMaxSeconds"),
                "idempotentReplay": False,
            }
        except QueueFullError as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(429, str(exc)) from exc
        except ReconstructionCancelled as exc:
            manager.fail_reserved(job_id, "Cancellation acknowledged during upload")
            raise HTTPException(409, "reconstruction upload was cancelled") from exc
        except HTTPException:
            manager.fail_reserved(job_id, "Provider input validation failed")
            raise
        except (TypeError, ValueError) as exc:
            manager.fail_reserved(job_id, f"Provider input validation failed: {exc}")
            raise HTTPException(400, str(exc)) from exc
        except Exception:
            manager.fail_reserved(job_id, "Provider input staging failed")
            LOGGER.exception("failed to stage NOCLIP job %s", job_id)
            raise HTTPException(500, "could not stage the reconstruction")

    @application.get("/v1/reconstructions/{job_id}", dependencies=[auth])
    async def noclip_reconstruction_status(job_id: str):
        if manager is None:
            raise HTTPException(503, "job manager is not ready")
        job = manager.get(job_id)
        if job is None or job.contract != NOCLIP_REQUEST_SCHEMA:
            raise HTTPException(404, "reconstruction not found")
        payload = manager.public(job_id) or {}
        status = {"complete": "completed", "running": "processing"}.get(
            job.status, job.status
        )
        return {
            "jobId": job.id,
            "status": status,
            "stage": payload.get("stage", "ready" if status == "completed" else status),
            "progress": job.progress / 100,
            "message": job.message,
            "metrics": job.summary or {},
            "queuePosition": payload.get("queuePosition"),
            "queueSize": payload.get("queueSize", 0),
            "activeJobs": payload.get("activeJobs", 0),
            "activeCapacity": payload.get("activeCapacity", 1),
            "availableCapacity": payload.get("availableCapacity", 0),
            "estimatedWaitMinSeconds": payload.get("estimatedWaitMinSeconds"),
            "estimatedWaitMaxSeconds": payload.get("estimatedWaitMaxSeconds"),
            "cancellationRequestedAt": job.cancellation_requested_at,
            "cancellationAcknowledgedAt": job.cancellation_acknowledged_at,
        }

    @application.delete("/v1/reconstructions/{job_id}", dependencies=[auth], status_code=202)
    async def cancel_noclip_reconstruction(job_id: str):
        if manager is None:
            raise HTTPException(503, "job manager is not ready")
        job = manager.get(job_id)
        if job is None or job.contract != NOCLIP_REQUEST_SCHEMA:
            raise HTTPException(404, "reconstruction not found")
        if job.status in JobManager.ACTIVE_STATUSES:
            manager.cancel(job_id)
        current = manager.get(job_id) or job
        return {
            "jobId": current.id,
            "status": current.status,
            "stage": current.stage,
            "cancellationRequestedAt": current.cancellation_requested_at,
            "cancellationAcknowledgedAt": current.cancellation_acknowledged_at,
        }

    @application.get("/v1/reconstructions/{job_id}/result", dependencies=[auth])
    async def noclip_reconstruction_result(job_id: str):
        if manager is None:
            raise HTTPException(503, "job manager is not ready")
        job = manager.get(job_id)
        if job is None or job.contract != NOCLIP_REQUEST_SCHEMA:
            raise HTTPException(404, "reconstruction not found")
        if job.status != "complete":
            raise HTTPException(409, f"reconstruction is {job.status}")
        result_dir = settings.jobs_dir / job_id / "result"
        paths = {
            "glb": result_dir / "reconstruction.glb",
            "summary": result_dir / "summary.json",
            "trajectory": result_dir / "trajectory.json",
            "pointCloud": result_dir / "point-cloud-lod.ply",
            "intrinsics": result_dir / "intrinsics.json",
            "diagnostics": result_dir / "quality-diagnostics.json",
            "manifest": result_dir / "reconstruction-manifest.json",
        }
        if any(not paths[name].is_file() for name in ("glb", "summary", "trajectory")):
            raise HTTPException(500, "reconstruction contract artifacts are incomplete")
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        trajectory = json.loads(paths["trajectory"].read_text(encoding="utf-8"))
        base = f"/v1/reconstructions/{job_id}/artifacts"
        artifacts = [
            {
                "kind": "reconstruction_glb",
                "url": f"{base}/reconstruction.glb",
                "fileName": "reconstruction.glb",
                "mimeType": "model/gltf-binary",
                "metadata": {
                    "framesUsed": summary.get("frames_used"),
                    "inferenceSeconds": summary.get("inference_seconds"),
                    "resultBytes": summary.get("result_bytes"),
                },
            },
            {
                "kind": "trajectory",
                "url": f"{base}/trajectory.json",
                "fileName": "trajectory.json",
                "mimeType": "application/json",
                "metadata": {"frameCount": len(trajectory.get("frames") or [])},
            },
        ]
        if paths["pointCloud"].is_file():
            artifacts.append(
                {
                    "kind": "point_cloud",
                    "url": f"{base}/point-cloud-lod.ply",
                    "fileName": "point-cloud-lod.ply",
                    "mimeType": "application/octet-stream",
                    "metadata": {
                        "points": summary.get("point_cloud_lod_points"),
                        "bytes": summary.get("point_cloud_lod_bytes"),
                        "coordinateFrame": "exported_lingbot_model",
                    },
                }
            )
        if paths["manifest"].is_file():
            reconstruction_manifest = json.loads(
                paths["manifest"].read_text(encoding="utf-8")
            )
            artifacts.append(
                {
                    "kind": "manifest",
                    "url": f"{base}/reconstruction-manifest.json",
                    "fileName": "reconstruction-manifest.json",
                    "mimeType": "application/json",
                    "metadata": {
                        "schema": reconstruction_manifest.get("schema"),
                        "coordinateContract": reconstruction_manifest.get(
                            "coordinateContract", {}
                        ),
                    },
                }
            )
        else:
            artifacts.append(
                {
                    "kind": "manifest",
                    "url": f"{base}/summary.json",
                    "fileName": "summary.json",
                    "mimeType": "application/json",
                    "metadata": {"legacy": True},
                }
            )
        if paths["diagnostics"].is_file():
            artifacts.append(
                {
                    "kind": "confidence",
                    "url": f"{base}/quality-diagnostics.json",
                    "fileName": "quality-diagnostics.json",
                    "mimeType": "application/json",
                    "metadata": {"schema": "noclip.lingbot.quality/1.0"},
                }
            )
        return {
            "schema": NOCLIP_RESULT_SCHEMA,
            "artifacts": artifacts,
            "frames": trajectory.get("frames") or [],
            "summary": summary,
        }

    @application.get(
        "/v1/reconstructions/{job_id}/artifacts/{file_name}", dependencies=[auth]
    )
    async def noclip_reconstruction_artifact(job_id: str, file_name: str):
        if manager is None:
            raise HTTPException(503, "job manager is not ready")
        job = manager.get(job_id)
        if job is None or job.contract != NOCLIP_REQUEST_SCHEMA:
            raise HTTPException(404, "reconstruction not found")
        safe_name = Path(file_name).name
        if safe_name not in {
            "reconstruction.glb",
            "point-cloud-lod.ply",
            "trajectory.json",
            "intrinsics.json",
            "quality-diagnostics.json",
            "reconstruction-manifest.json",
            "summary.json",
        }:
            raise HTTPException(404, "artifact not found")
        path = settings.jobs_dir / job_id / "result" / safe_name
        if not path.is_file():
            raise HTTPException(404, "artifact not found")
        media_type = {
            ".glb": "model/gltf-binary",
            ".ply": "application/octet-stream",
        }.get(path.suffix, "application/json")
        return FileResponse(path, media_type=media_type, filename=safe_name)

    assets = settings.web_dist / "assets"
    if assets.is_dir():
        application.mount("/assets", StaticFiles(directory=assets), name="assets")

    @application.get("/", include_in_schema=False)
    async def index():
        path = settings.web_dist / "index.html"
        if not path.is_file():
            raise HTTPException(503, "frontend assets have not been built")
        return FileResponse(path, media_type="text/html")

    return application


app = create_app()
