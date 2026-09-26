from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .broker import GpuLease
from .artifacts import (
    artifact_record,
    write_intrinsics,
    write_json,
    write_point_cloud_lod,
    write_reconstruction_manifest,
)
from .config import Settings
from .noclip_contract import exported_model_transform
from .weights import WeightSpec, pull_weights


LOGGER = logging.getLogger(__name__)


def _current_rss_bytes() -> int | None:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


class InferenceEngine:
    """One resident LingBot-Map model with serialized inference."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lease = GpuLease.from_env(required=settings.require_gpu_broker)
        self._lock = threading.Lock()
        self.model: Any = None
        self.device: Any = None
        self.dtype: Any = None
        self.weight_path: Path | None = None
        self.load_summary: dict[str, Any] = {}
        self.ready = False

    def load(self) -> dict[str, Any]:
        import torch

        from lingbot_map.models.gct_stream import GCTStream

        if self.settings.require_gpu_broker and not self.lease.configured:
            raise RuntimeError(
                "GPU broker lease is required; launch through scripts/deploy.sh"
            )

        self.weight_path = pull_weights(WeightSpec.from_settings(self.settings))
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.empty(1, device=self.device)
        elif self.settings.allow_cpu:
            self.device = torch.device("cpu")
        else:
            raise RuntimeError("CUDA is unavailable and LINGBOT_ALLOW_CPU is disabled")

        started = time.time()
        model = GCTStream(
            img_size=self.settings.image_size,
            patch_size=self.settings.patch_size,
            enable_3d_rope=True,
            max_frame_num=self.settings.max_frame_num,
            kv_cache_sliding_window=64,
            kv_cache_scale_frames=8,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=self.settings.use_sdpa,
            camera_num_iterations=self.settings.camera_iterations,
        )
        checkpoint = torch.load(self.weight_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        del checkpoint, state_dict

        model = model.to(self.device).eval()
        if self.device.type == "cuda":
            major, _minor = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if major >= 8 else torch.float16
            if getattr(model, "aggregator", None) is not None:
                model.aggregator = model.aggregator.to(dtype=self.dtype)
            torch.cuda.synchronize()
        else:
            self.dtype = torch.float32

        self.model = model
        self.load_summary = {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "weight": self.weight_path.name,
            "weight_bytes": self.weight_path.stat().st_size,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "load_seconds": round(time.time() - started, 3),
        }
        self.ready = True
        LOGGER.info("model ready: %s", self.load_summary)
        return dict(self.load_summary)

    def reconstruct(
        self,
        *,
        image_dir: Path,
        result_dir: Path,
        num_scale_frames: int,
        keyframe_interval: int,
        confidence_percentile: float,
        include_cameras: bool,
        input_summary: dict[str, Any],
        noclip_manifest: dict[str, Any] | None = None,
        progress_callback: Callable[..., None] | None = None,
        cancellation_checkpoint: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        import torch

        from demo import load_images, postprocess, prepare_for_visualization
        from lingbot_map.vis.glb_export import predictions_to_glb
        from .noclip_contract import build_aligned_camera_frames, write_noclip_trajectory

        if not self.ready or self.model is None:
            raise RuntimeError("model is not ready")

        result_dir.mkdir(parents=True, exist_ok=True)

        def checkpoint() -> None:
            if cancellation_checkpoint is not None:
                cancellation_checkpoint()

        def report(percent: int, message: str, stage: str) -> None:
            if progress_callback is not None:
                progress_callback(percent, message, stage)

        with self._lock:
            images = None
            predictions = None
            prepared_for_export = None
            rss_before_bytes = _current_rss_bytes()
            gpu_prepare_started = time.monotonic()
            gpu_cleanup_seconds = 0.0
            checkpoint()
            report(32, "Preparing broker-scoped GPU capacity", "preparing_gpu")
            self.lease.prepare()
            gpu_prepare_seconds = time.monotonic() - gpu_prepare_started
            try:
                checkpoint()
                report(38, "Loading the bounded sampled frames", "preprocessing")
                images, paths, _ = load_images(
                    image_folder=str(image_dir),
                    first_k=self.settings.max_frames,
                    image_size=self.settings.image_size,
                    patch_size=self.settings.patch_size,
                )
                if len(paths) < 2:
                    raise ValueError("at least two frames are required")
                checkpoint()

                frame_count = int(images.shape[0])
                scale_frames = min(max(1, num_scale_frames), frame_count)
                keyframe_interval = max(1, keyframe_interval)
                images = images.to(self.device)

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                autocast = (
                    torch.amp.autocast("cuda", dtype=self.dtype)
                    if self.device.type == "cuda"
                    else contextlib.nullcontext()
                )
                report(
                    48,
                    (
                        f"Reconstructing {frame_count} frames; cancellation waits for "
                        "the current upstream model call to return"
                    ),
                    "inference",
                )
                started = time.time()
                with torch.no_grad(), autocast:
                    predictions = self.model.inference_streaming(
                        images,
                        num_scale_frames=scale_frames,
                        keyframe_interval=keyframe_interval,
                        output_device=torch.device("cpu"),
                    )
                inference_seconds = time.time() - started
                checkpoint()

                report(80, "Postprocessing solved geometry", "postprocessing")
                images_for_post = predictions["images"]
                del images
                images = None
                predictions, images_cpu = postprocess(predictions, images_for_post)
                checkpoint()
                prepared_for_export = prepare_for_visualization(predictions, images_cpu)
                peak_memory_gb = (
                    round(torch.cuda.max_memory_allocated() / 1e9, 3)
                    if self.device.type == "cuda"
                    else None
                )
            finally:
                cleanup_started = time.monotonic()
                try:
                    if self.model is not None and hasattr(self.model, "clean_kv_cache"):
                        self.model.clean_kv_cache()
                finally:
                    try:
                        if self.device is not None and self.device.type == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    finally:
                        self.lease.ready()
                        gpu_cleanup_seconds = time.monotonic() - cleanup_started

            if prepared_for_export is None:
                raise RuntimeError("inference did not produce exportable predictions")

            checkpoint()
            glb_path = result_dir / "reconstruction.glb"
            report(86, "Exporting normalized model geometry", "export")
            scene = predictions_to_glb(
                prepared_for_export,
                conf_thres=float(confidence_percentile),
                show_cam=bool(include_cameras),
                target_dir=str(result_dir),
                mask_sky=False,
            )
            scene.export(glb_path)
            checkpoint()
            point_cloud_path = result_dir / "point-cloud-lod.ply"
            report(91, "Exporting the bounded point-cloud LOD", "export")
            point_cloud_quality = write_point_cloud_lod(
                prepared_for_export,
                point_cloud_path,
                confidence_percentile=float(confidence_percentile),
                max_points=self.settings.point_cloud_max_points,
            )
            checkpoint()
            report(95, "Writing solved trajectory and diagnostics", "export")

            noclip_frames: list[dict[str, Any]] = []
            trajectory_path: Path | None = None
            intrinsics_path: Path | None = None
            intrinsics_payload: dict[str, Any] | None = None
            reconstruction_manifest_path: Path | None = None
            if noclip_manifest is not None:
                noclip_frames = build_aligned_camera_frames(
                    prepared_for_export, noclip_manifest, input_summary
                )
                trajectory_path = write_noclip_trajectory(
                    result_dir,
                    capture_session_id=str(noclip_manifest["captureSessionId"]),
                    frames=noclip_frames,
                )
                intrinsics_path = result_dir / "intrinsics.json"
                intrinsics_payload = write_intrinsics(
                    intrinsics_path,
                    capture_session_id=str(noclip_manifest["captureSessionId"]),
                    frames=noclip_frames,
                )

            rss_after_cleanup_bytes = _current_rss_bytes()
            quality = {
                "schema": "noclip.lingbot.quality/1.0",
                "coordinateFrame": "exported_lingbot_model",
                "frameCount": frame_count,
                "pointCloud": point_cloud_quality,
                "confidenceFiltering": {
                    "enabled": True,
                    "percentile": float(confidence_percentile),
                },
                "skyMasking": {
                    "enabled": False,
                    "reason": (
                        "disabled pending representative outdoor field-corpus "
                        "accuracy evaluation"
                    ),
                },
                "memory": {
                    "rssBeforeBytes": rss_before_bytes,
                    "rssAfterGpuCleanupBytes": rss_after_cleanup_bytes,
                    "rssGrowthBytes": (
                        rss_after_cleanup_bytes - rss_before_bytes
                        if rss_before_bytes is not None
                        and rss_after_cleanup_bytes is not None
                        else None
                    ),
                    "peakGpuMemoryGb": peak_memory_gb,
                },
                "gpuBroker": {
                    "prepareSeconds": round(gpu_prepare_seconds, 3),
                    "cleanupToReadySeconds": round(gpu_cleanup_seconds, 3),
                },
            }
            diagnostics_path = result_dir / "quality-diagnostics.json"
            write_json(diagnostics_path, quality)
            summary = {
                "frames_used": frame_count,
                "num_scale_frames": scale_frames,
                "keyframe_interval": keyframe_interval,
                "confidence_percentile": float(confidence_percentile),
                "include_cameras": bool(include_cameras),
                "inference_seconds": round(inference_seconds, 3),
                "peak_gpu_memory_gb": peak_memory_gb,
                "rss_before_bytes": rss_before_bytes,
                "rss_after_gpu_cleanup_bytes": rss_after_cleanup_bytes,
                "rss_growth_bytes": quality["memory"]["rssGrowthBytes"],
                "gpu_prepare_seconds": round(gpu_prepare_seconds, 3),
                "gpu_cleanup_to_ready_seconds": round(gpu_cleanup_seconds, 3),
                "result_bytes": glb_path.stat().st_size,
                "point_cloud_lod_bytes": point_cloud_path.stat().st_size,
                "point_cloud_lod_points": point_cloud_quality["exportedPoints"],
                "model": dict(self.load_summary),
                "input": input_summary,
                "noclip": {
                    "capture_session_id": noclip_manifest.get("captureSessionId"),
                    "trajectory_frames": len(noclip_frames),
                    "trajectory_bytes": trajectory_path.stat().st_size,
                } if noclip_manifest is not None and trajectory_path is not None else None,
            }
            (result_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if noclip_manifest is not None and trajectory_path is not None and intrinsics_path is not None:
                artifact_records = [
                    artifact_record(
                        glb_path,
                        kind="reconstruction_glb",
                        mime_type="model/gltf-binary",
                    ),
                    artifact_record(
                        point_cloud_path,
                        kind="point_cloud",
                        mime_type="application/octet-stream",
                    ),
                    artifact_record(
                        trajectory_path,
                        kind="trajectory",
                        mime_type="application/json",
                    ),
                    artifact_record(
                        intrinsics_path,
                        kind="intrinsics",
                        mime_type="application/json",
                    ),
                    artifact_record(
                        diagnostics_path,
                        kind="confidence",
                        mime_type="application/json",
                    ),
                ]
                reconstruction_manifest_path = result_dir / "reconstruction-manifest.json"
                write_reconstruction_manifest(
                    reconstruction_manifest_path,
                    capture_session_id=str(noclip_manifest["captureSessionId"]),
                    model=dict(self.load_summary),
                    artifacts=artifact_records,
                    transform=exported_model_transform(prepared_for_export),
                    quality=quality,
                    intrinsics=intrinsics_payload,
                )
                summary["noclip"]["manifest_bytes"] = (
                    reconstruction_manifest_path.stat().st_size
                )
                (result_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            checkpoint()
            report(99, "Provider artifacts are durable", "artifact_upload")
            return summary
