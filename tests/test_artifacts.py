from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from webapp.artifacts import (
    RECONSTRUCTION_MANIFEST_SCHEMA,
    artifact_record,
    write_point_cloud_lod,
    write_reconstruction_manifest,
)
from webapp.noclip_contract import exported_model_transform


PLY_VERTEX = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def _predictions() -> dict:
    return {
        "world_points": np.array(
            [
                [[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]],
                [[[12.0, 0.0, 0.0], [13.0, 0.0, 0.0]]],
            ]
        ),
        "world_points_conf": np.array([[[1.0, 2.0]], [[3.0, 4.0]]]),
        "images": np.array(
            [
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                [[[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]],
            ]
        ),
        "extrinsic": np.array(
            [
                [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0]],
                [[1, 0, 0, 11], [0, 1, 0, 0], [0, 0, 1, 0]],
            ],
            dtype=np.float64,
        ),
        "intrinsic": np.repeat(np.eye(3)[None], 2, axis=0),
    }


def test_point_cloud_lod_is_bounded_and_uses_the_exported_model_transform(tmp_path):
    predictions = _predictions()
    path = tmp_path / "point-cloud-lod.ply"
    quality = write_point_cloud_lod(
        predictions,
        path,
        confidence_percentile=0,
        max_points=2,
    )
    payload = path.read_bytes()
    marker = payload.index(b"end_header\n") + len(b"end_header\n")
    vertices = np.frombuffer(payload[marker:], dtype=PLY_VERTEX)
    source = predictions["world_points"].reshape(-1, 3)[[0, 3]]
    transform = exported_model_transform(predictions)
    expected = source @ transform[:3, :3].T + transform[:3, 3]

    assert quality["sourcePoints"] == 4
    assert quality["exportedPoints"] == 2
    assert quality["geometrySource"] == "world_points"
    assert np.allclose(
        np.column_stack((vertices["x"], vertices["y"], vertices["z"])),
        expected,
    )
    assert vertices[["red", "green", "blue"]].tolist() == [
        (255, 0, 0),
        (255, 255, 255),
    ]


def test_point_cloud_lod_backprojects_depth_when_point_head_is_absent(tmp_path):
    predictions = {
        "depth": np.array([[[[2.0], [3.0]]]]),
        "depth_conf": np.array([[[1.0, 2.0]]]),
        "images": np.array([[[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]]),
        "extrinsic": np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ]
        ),
        "intrinsic": np.array(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        ),
    }
    path = tmp_path / "point-cloud-lod.ply"

    quality = write_point_cloud_lod(
        predictions,
        path,
        confidence_percentile=0,
        max_points=10,
    )
    payload = path.read_bytes()
    marker = payload.index(b"end_header\n") + len(b"end_header\n")
    vertices = np.frombuffer(payload[marker:], dtype=PLY_VERTEX)
    camera_points = np.array([[0.0, 0.0, 2.0], [3.0, 0.0, 3.0]])
    transform = exported_model_transform(predictions)
    expected = camera_points @ transform[:3, :3].T + transform[:3, 3]

    assert quality["geometrySource"] == "depth_backprojection"
    assert quality["sourcePoints"] == 2
    assert quality["exportedPoints"] == 2
    assert np.allclose(
        np.column_stack((vertices["x"], vertices["y"], vertices["z"])),
        expected,
    )
    assert vertices[["red", "green", "blue"]].tolist() == [
        (255, 0, 0),
        (0, 0, 255),
    ]


def test_versioned_manifest_records_artifact_integrity_and_one_frame(tmp_path):
    artifact = tmp_path / "reconstruction.glb"
    artifact.write_bytes(b"glTF")
    record = artifact_record(
        artifact, kind="reconstruction_glb", mime_type="model/gltf-binary"
    )
    manifest_path = tmp_path / "reconstruction-manifest.json"
    write_reconstruction_manifest(
        manifest_path,
        capture_session_id="capture-1",
        model={"weight": "fixture.pt"},
        artifacts=[record],
        transform=np.eye(4),
        quality={"schema": "noclip.lingbot.quality/1.0"},
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema"] == RECONSTRUCTION_MANIFEST_SCHEMA
    assert payload["coordinateContract"]["modelFrame"] == "exported_lingbot_model"
    assert payload["artifacts"][0]["sha256"] == (
        "c74f919439792582aa4f0b188ec2a928675cdb3ba72797781ceb6dfaa86b313f"
    )
