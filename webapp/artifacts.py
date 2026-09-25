from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .noclip_contract import exported_model_transform


RECONSTRUCTION_MANIFEST_SCHEMA = "noclip.lingbot.reconstruction/1.0"


def _images_as_rgb(predictions: dict[str, Any]) -> np.ndarray:
    images = np.asarray(predictions.get("images"))
    if images.ndim != 4:
        raise ValueError("LingBot predictions are missing frame colors")
    if images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    if images.shape[-1] != 3:
        raise ValueError("LingBot frame colors must have three channels")
    if np.issubdtype(images.dtype, np.floating):
        images = np.clip(images, 0.0, 1.0) * 255.0
    return np.asarray(images, dtype=np.uint8)


def write_point_cloud_lod(
    predictions: dict[str, Any],
    path: Path,
    *,
    confidence_percentile: float,
    max_points: int,
) -> dict[str, Any]:
    """Write one deterministic bounded binary PLY in the exported GLB frame."""

    points = np.asarray(predictions.get("world_points"), dtype=np.float64)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("LingBot predictions are missing point-map geometry")
    confidence_value = predictions.get("world_points_conf")
    confidence = (
        np.asarray(confidence_value, dtype=np.float64)
        if confidence_value is not None
        else np.ones(points.shape[:-1], dtype=np.float64)
    )
    colors = _images_as_rgb(predictions)
    if confidence.shape != points.shape[:-1] or colors.shape[:-1] != points.shape[:-1]:
        raise ValueError("point, confidence, and color grids do not share one frame")

    flat_points = points.reshape(-1, 3)
    flat_confidence = confidence.reshape(-1)
    flat_colors = colors.reshape(-1, 3)
    finite = np.isfinite(flat_points).all(axis=1) & np.isfinite(flat_confidence)
    finite_confidence = flat_confidence[finite]
    if not len(finite_confidence):
        raise ValueError("point cloud contains no finite samples")
    threshold = (
        float(np.percentile(finite_confidence, confidence_percentile))
        if confidence_percentile > 0
        else 0.0
    )
    eligible = np.flatnonzero(
        finite & (flat_confidence >= threshold) & (flat_confidence > 1e-5)
    )
    if not len(eligible):
        raise ValueError("confidence filtering removed every point")
    if len(eligible) > max_points:
        positions = np.linspace(0, len(eligible) - 1, max_points, dtype=np.int64)
        selected = eligible[positions]
    else:
        selected = eligible

    transform = exported_model_transform(predictions)
    exported = (
        flat_points[selected] @ transform[:3, :3].T + transform[:3, 3]
    ).astype("<f4", copy=False)
    exported_colors = flat_colors[selected]
    vertices = np.empty(
        len(selected),
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    vertices["x"], vertices["y"], vertices["z"] = exported.T
    vertices["red"], vertices["green"], vertices["blue"] = exported_colors.T
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment coordinate_frame exported_lingbot_model\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)

    lower = exported.min(axis=0).astype(float).tolist()
    upper = exported.max(axis=0).astype(float).tolist()
    return {
        "schema": "noclip.lingbot.point-cloud-lod/1.0",
        "coordinateFrame": "exported_lingbot_model",
        "sourcePoints": int(len(flat_points)),
        "eligiblePoints": int(len(eligible)),
        "exportedPoints": int(len(exported)),
        "maxPoints": int(max_points),
        "confidencePercentile": float(confidence_percentile),
        "confidenceThreshold": threshold,
        "boundingBoxM": {"minimum": lower, "maximum": upper},
    }


def write_intrinsics(
    path: Path,
    *,
    capture_session_id: str,
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema": "noclip.lingbot.intrinsics/1.0",
        "captureSessionId": capture_session_id,
        "units": "pixels",
        "frames": [
            {
                "sequenceNumber": frame.get("sequenceNumber"),
                "sensorSampleIndex": frame.get("sensorSampleIndex"),
                "intrinsics": frame.get("intrinsics"),
            }
            for frame in frames
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def artifact_record(path: Path, *, kind: str, mime_type: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "kind": kind,
        "fileName": path.name,
        "mimeType": mime_type,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "coordinateFrame": "exported_lingbot_model",
    }


def write_reconstruction_manifest(
    path: Path,
    *,
    capture_session_id: str,
    model: dict[str, Any],
    artifacts: list[dict[str, Any]],
    transform: np.ndarray,
    quality: dict[str, Any],
    intrinsics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": RECONSTRUCTION_MANIFEST_SCHEMA,
        "captureSessionId": capture_session_id,
        "coordinateContract": {
            "modelFrame": "exported_lingbot_model",
            "cameraAxes": "opencv_x_right_y_down_z_forward",
            "quaternionOrder": "xyzw",
            "units": "meters",
            "transformConvention": "column_vector_model_from_lingbot_local",
            "modelFromLingbotLocal": transform.astype(float).tolist(),
        },
        "model": model,
        "artifacts": artifacts,
        "intrinsics": intrinsics,
        "quality": quality,
    }
    write_json(path, payload)
    return payload
