from __future__ import annotations

import numpy as np
import pytest

from webapp.noclip_contract import (
    NOCLIP_REQUEST_SCHEMA,
    build_aligned_camera_frames,
    validate_noclip_manifest,
)


def manifest() -> dict:
    return {
        "schema": NOCLIP_REQUEST_SCHEMA,
        "captureSessionId": "capture-1",
        "coordinateSystem": {
            "geodetic": "WGS84",
            "localFrame": "ENU",
            "cameraAxes": "opencv_x_right_y_down_z_forward",
            "quaternionOrder": "xyzw",
            "units": "meters",
        },
        "media": [
            {"sequenceNumber": 4, "monotonicMs": 100.0},
            {"sequenceNumber": 5, "monotonicMs": 200.0},
        ],
        "poses": [
            {"sampleIndex": 10, "monotonicMs": 100.0},
            {"sampleIndex": 11, "monotonicMs": 200.0},
        ],
    }


def test_manifest_requires_the_exact_grounding_frame_contract():
    assert validate_noclip_manifest(manifest())["captureSessionId"] == "capture-1"
    invalid = manifest()
    invalid["coordinateSystem"]["localFrame"] = "NED"
    with pytest.raises(ValueError, match="localFrame"):
        validate_noclip_manifest(invalid)


def test_trajectory_uses_the_same_first_camera_alignment_as_glb_export():
    extrinsics = np.array(
        [
            [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0]],
            [[1, 0, 0, 8], [0, 1, 0, 0], [0, 0, 1, 0]],
        ],
        dtype=np.float64,
    )
    frames = build_aligned_camera_frames(
        {"extrinsic": extrinsics, "intrinsic": np.repeat(np.eye(3)[None], 2, axis=0)},
        manifest(),
    )
    assert frames[0]["sensorSampleIndex"] == 10
    assert frames[1]["sequenceNumber"] == 5
    assert np.allclose(frames[0]["cameraToWorld"]["positionM"], [0, 0, 0])
    assert np.isclose(
        np.linalg.norm(frames[1]["cameraToWorld"]["positionM"]), 2.0
    )
