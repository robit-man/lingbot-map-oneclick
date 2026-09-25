from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


NOCLIP_REQUEST_SCHEMA = "noclip.lingbot.request/1.0"
NOCLIP_RESULT_SCHEMA = "noclip.lingbot.result/1.0"


def exported_model_transform(predictions: dict[str, Any]) -> np.ndarray:
    """Return the transform applied to both GLB geometry and solved cameras."""

    extrinsics = np.asarray(predictions.get("extrinsic"), dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4) or not len(extrinsics):
        raise ValueError("LingBot predictions are missing [frames,3,4] extrinsics")
    first = np.eye(4, dtype=np.float64)
    first[:3, :4] = extrinsics[0]
    opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    align_y_180 = np.eye(4, dtype=np.float64)
    align_y_180[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    return np.linalg.inv(first) @ opengl @ align_y_180


def validate_noclip_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != NOCLIP_REQUEST_SCHEMA:
        raise ValueError(f"manifest schema must be {NOCLIP_REQUEST_SCHEMA}")
    session_id = value.get("captureSessionId")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("manifest captureSessionId is required")
    coordinate_system = value.get("coordinateSystem")
    if not isinstance(coordinate_system, dict):
        raise ValueError("manifest coordinateSystem is required")
    required = {
        "geodetic": "WGS84",
        "localFrame": "ENU",
        "cameraAxes": "opencv_x_right_y_down_z_forward",
        "quaternionOrder": "xyzw",
        "units": "meters",
    }
    for field, expected in required.items():
        if coordinate_system.get(field) != expected:
            raise ValueError(f"coordinateSystem.{field} must be {expected}")
    if not isinstance(value.get("media"), list):
        raise ValueError("manifest media must be an array")
    if not isinstance(value.get("poses"), list):
        raise ValueError("manifest poses must be an array")
    return value


def _pose_for_frame(
    index: int,
    frame_count: int,
    manifest: dict[str, Any],
) -> tuple[int, float | None, int]:
    poses = manifest.get("poses") or []
    media = manifest.get("media") or []
    target: float | None = None
    sequence_number = index
    if len(media) == frame_count and index < len(media):
        entry = media[index]
        sequence_number = int(entry.get("sequenceNumber", index))
        value = entry.get("monotonicMs")
        target = float(value) if isinstance(value, (int, float)) else None
    elif len(media) == 1:
        entry = media[0]
        sequence_number = int(entry.get("sequenceNumber", 0)) + index
        start = entry.get("monotonicMs")
        duration = entry.get("durationMs")
        if isinstance(start, (int, float)) and isinstance(duration, (int, float)):
            target = float(start) + float(duration) * index / max(1, frame_count - 1)
    if not poses:
        return index, target, sequence_number
    if target is None:
        pose_index = round(index * max(0, len(poses) - 1) / max(1, frame_count - 1))
        value = poses[pose_index].get("monotonicMs")
        target = float(value) if isinstance(value, (int, float)) else None
    else:
        pose_index = min(
            range(len(poses)),
            key=lambda candidate: abs(
                float(poses[candidate].get("monotonicMs", target)) - target
            ),
        )
    sample_index = int(poses[pose_index].get("sampleIndex", pose_index))
    return sample_index, target, sequence_number


def build_aligned_camera_frames(
    predictions: dict[str, Any],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return camera poses in the exact local frame exported by the GLB."""

    extrinsics = np.asarray(predictions.get("extrinsic"), dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4) or not len(extrinsics):
        raise ValueError("LingBot predictions are missing [frames,3,4] extrinsics")
    intrinsics_value = predictions.get("intrinsic")
    intrinsics = (
        np.asarray(intrinsics_value, dtype=np.float64)
        if intrinsics_value is not None
        else None
    )

    full = np.repeat(np.eye(4, dtype=np.float64)[None, ...], len(extrinsics), axis=0)
    full[:, :3, :4] = extrinsics
    scene_alignment = exported_model_transform(predictions)

    frames: list[dict[str, Any]] = []
    for index, world_to_camera in enumerate(full):
        camera_to_world = scene_alignment @ np.linalg.inv(world_to_camera)
        sample_index, monotonic_ms, sequence_number = _pose_for_frame(
            index, len(full), manifest
        )
        frame: dict[str, Any] = {
            "sequenceNumber": sequence_number,
            "sensorSampleIndex": sample_index,
            "monotonicMs": monotonic_ms,
            "cameraToWorld": {
                "positionM": camera_to_world[:3, 3].tolist(),
                "quaternionXyzw": Rotation.from_matrix(
                    camera_to_world[:3, :3]
                ).as_quat().tolist(),
            },
        }
        if intrinsics is not None and index < len(intrinsics):
            matrix = intrinsics[index]
            frame["intrinsics"] = {
                "fx": float(matrix[0, 0]),
                "fy": float(matrix[1, 1]),
                "cx": float(matrix[0, 2]),
                "cy": float(matrix[1, 2]),
            }
        frames.append(frame)
    return frames


def write_noclip_trajectory(
    result_dir: Path,
    *,
    capture_session_id: str,
    frames: list[dict[str, Any]],
) -> Path:
    path = result_dir / "trajectory.json"
    path.write_text(
        json.dumps(
            {
                "schema": "noclip.lingbot.trajectory/1.0",
                "captureSessionId": capture_session_id,
                "coordinateFrame": "exported_lingbot_model",
                "cameraAxes": "opencv_x_right_y_down_z_forward",
                "quaternionOrder": "xyzw",
                "units": "meters",
                "frames": frames,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
