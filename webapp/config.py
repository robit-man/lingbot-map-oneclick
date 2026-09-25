from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    api_token: str
    model_dir: Path
    model_repo: str
    model_filename: str
    model_revision: str
    model_sha256: str
    model_size_bytes: int
    model_path: str
    jobs_dir: Path
    web_dist: Path
    max_upload_bytes: int
    max_files: int
    max_frames: int
    max_queue: int
    retain_jobs: int
    point_cloud_max_points: int
    image_size: int
    patch_size: int
    max_frame_num: int
    default_scale_frames: int
    camera_iterations: int
    use_sdpa: bool
    allow_cpu: bool
    require_gpu_broker: bool

    @classmethod
    def from_env(cls, *, require_token: bool = True) -> "Settings":
        repo_root = Path(__file__).resolve().parents[1]
        api_token = os.getenv("LINGBOT_API_TOKEN", "").strip()
        if require_token and len(api_token) < 24:
            raise ValueError(
                "LINGBOT_API_TOKEN must be set to at least 24 characters; "
                "scripts/deploy.sh generates one automatically"
            )

        model_repo = os.getenv("LINGBOT_MODEL_REPO", "robbyant/lingbot-map")
        model_filename = os.getenv("LINGBOT_MODEL_FILENAME", "lingbot-map.pt")
        model_revision = os.getenv(
            "LINGBOT_MODEL_REVISION",
            "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9",
        )
        is_default_checkpoint = (
            model_repo == "robbyant/lingbot-map"
            and model_filename == "lingbot-map.pt"
            and model_revision == "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9"
        )
        default_sha = (
            "ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72"
            if is_default_checkpoint
            else ""
        )
        default_size = 4_632_303_465 if is_default_checkpoint else 0

        return cls(
            api_token=api_token,
            model_dir=Path(os.getenv("LINGBOT_MODEL_DIR", "/models")).resolve(),
            model_repo=model_repo,
            model_filename=model_filename,
            model_revision=model_revision,
            model_sha256=os.getenv("LINGBOT_MODEL_SHA256", default_sha).strip().lower(),
            model_size_bytes=_env_int(
                "LINGBOT_MODEL_SIZE_BYTES", default_size, minimum=0
            ),
            model_path=os.getenv("LINGBOT_MODEL_PATH", "").strip(),
            jobs_dir=Path(os.getenv("LINGBOT_JOBS_DIR", "/data/jobs")).resolve(),
            web_dist=Path(
                os.getenv("LINGBOT_WEB_DIST", str(repo_root / "web" / "dist"))
            ).resolve(),
            max_upload_bytes=_env_int(
                "LINGBOT_MAX_UPLOAD_BYTES", 90 * 1024 * 1024
            ),
            max_files=_env_int("LINGBOT_MAX_FILES", 160),
            max_frames=_env_int("LINGBOT_MAX_FRAMES", 96, minimum=2),
            max_queue=_env_int("LINGBOT_MAX_QUEUE", 4),
            retain_jobs=_env_int("LINGBOT_RETAIN_JOBS", 20),
            point_cloud_max_points=_env_int(
                "LINGBOT_POINT_CLOUD_MAX_POINTS", 250_000, minimum=1_000
            ),
            image_size=_env_int("LINGBOT_IMAGE_SIZE", 518),
            patch_size=_env_int("LINGBOT_PATCH_SIZE", 14),
            max_frame_num=_env_int("LINGBOT_MAX_FRAME_NUM", 1024),
            default_scale_frames=_env_int("LINGBOT_SCALE_FRAMES", 4),
            camera_iterations=_env_int("LINGBOT_CAMERA_ITERATIONS", 1),
            use_sdpa=_env_bool("LINGBOT_USE_SDPA", True),
            allow_cpu=_env_bool("LINGBOT_ALLOW_CPU", False),
            require_gpu_broker=_env_bool("LINGBOT_REQUIRE_GPU_BROKER", True),
        )
