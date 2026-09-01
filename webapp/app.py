from __future__ import annotations

import asyncio
import hmac
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
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
    extract_video_frames,
    validate_image_files,
)
from .jobs import JobManager, QueueFullError


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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    engine = InferenceEngine(settings)
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
        x_api_token: Annotated[str | None, Header()] = None,
    ) -> None:
        bearer = ""
        if authorization and authorization.lower().startswith("bearer "):
            bearer = authorization[7:].strip()
        supplied = bearer or (x_api_token or "")
        if not supplied or not hmac.compare_digest(supplied, settings.api_token):
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

        job_id = uuid.uuid4().hex
        job_dir = settings.jobs_dir / job_id
        upload_dir = job_dir / "upload"
        try:
            uploaded = await _save_uploads(
                files,
                upload_dir,
                byte_limit=settings.max_upload_bytes,
                file_limit=settings.max_files,
            )
            image_uploads = [path for path in uploaded if path.suffix in IMAGE_SUFFIXES]
            video_uploads = [path for path in uploaded if path.suffix in VIDEO_SUFFIXES]
            if video_uploads and image_uploads:
                raise HTTPException(400, "upload either one video or an image sequence")
            if len(video_uploads) > 1:
                raise HTTPException(400, "upload only one video")

            image_dir = job_dir / "frames"
            if video_uploads:
                _paths, input_summary = await asyncio.to_thread(
                    extract_video_frames,
                    video_uploads[0],
                    image_dir,
                    fps=fps,
                    max_frames=max_frames,
                )
            else:
                if len(image_uploads) < 2:
                    raise HTTPException(400, "upload at least two images")
                selected = image_uploads[:max_frames]
                await asyncio.to_thread(validate_image_files, selected)
                image_dir.mkdir(parents=True, exist_ok=True)
                for index, source in enumerate(selected):
                    source.replace(image_dir / f"{index:06d}{source.suffix}")
                input_summary = {
                    "input_mode": "images",
                    "frames_used": len(selected),
                }

            if manager is None:
                raise HTTPException(503, "job manager is not ready")
            job = manager.submit(
                job_id,
                image_dir=image_dir,
                result_dir=job_dir / "result",
                num_scale_frames=num_scale_frames,
                keyframe_interval=keyframe_interval,
                confidence_percentile=confidence_percentile,
                include_cameras=include_cameras,
                input_summary=input_summary,
            )
            return job.public()
        except QueueFullError as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(429, str(exc)) from exc
        except HTTPException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        except ValueError as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(400, str(exc)) from exc
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            LOGGER.exception("failed to stage job %s", job_id)
            raise HTTPException(500, "could not stage the upload")

    @application.get("/api/jobs/{job_id}", dependencies=[auth])
    async def job_status(job_id: str):
        if manager is None:
            raise HTTPException(503, "job manager is not ready")
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job.public()

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
