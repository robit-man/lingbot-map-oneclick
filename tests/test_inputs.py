from pathlib import Path

import pytest
from PIL import Image

from webapp.inputs import (
    resolve_frame_presentation_timestamps,
    stage_input_media,
    validate_image_files,
)


def test_validate_image_files_accepts_decodable_images(tmp_path: Path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (32, 24), (10, 20, 30)).save(image_path)
    validate_image_files([image_path])


def test_validate_image_files_rejects_invalid_payload(tmp_path: Path):
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="invalid image upload"):
        validate_image_files([image_path])


def test_long_image_sequence_is_bounded_and_reports_preprocessing(tmp_path: Path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    paths = []
    for index in range(12):
        path = uploads / f"{index:03d}.png"
        Image.new("RGB", (8, 8), (index, index, index)).save(path)
        paths.append(path)
    progress = []
    result = stage_input_media(
        paths,
        tmp_path / "frames",
        fps=8,
        max_frames=4,
        cancellation_checkpoint=lambda: None,
        progress_callback=lambda percent, message, stage: progress.append(
            (percent, message, stage)
        ),
    )
    assert result["input_summary"]["frames_used"] == 4
    assert len(list((tmp_path / "frames").iterdir())) == 4
    assert {entry[2] for entry in progress} == {"preprocessing"}


def test_image_preprocessing_stops_at_a_cooperative_checkpoint(tmp_path: Path):
    paths = []
    for index in range(4):
        path = tmp_path / f"{index:03d}.png"
        Image.new("RGB", (8, 8), (index, 0, 0)).save(path)
        paths.append(path)
    calls = 0

    def checkpoint():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("cancelled fixture")

    with pytest.raises(RuntimeError, match="cancelled fixture"):
        stage_input_media(
            paths,
            tmp_path / "frames",
            fps=8,
            max_frames=4,
            cancellation_checkpoint=checkpoint,
            progress_callback=lambda *_args: None,
        )


def test_decoder_presentation_timestamps_preserve_variable_frame_rate():
    values, association, uncertainty = resolve_frame_presentation_timestamps(
        [0.0, 41.5, 109.25], [0, 1, 3], 30.0
    )
    assert values == [0.0, 41.5, 109.25]
    assert association == "decoded-frame-pts"
    assert uncertainty == 16.667


def test_invalid_decoder_timestamps_fall_back_with_explicit_uncertainty():
    values, association, uncertainty = resolve_frame_presentation_timestamps(
        [0.0, 0.0, 0.0], [0, 2, 4], 20.0
    )
    assert values == [0.0, 100.0, 200.0]
    assert association == "source-fps-fallback"
    assert uncertainty == 50.0
