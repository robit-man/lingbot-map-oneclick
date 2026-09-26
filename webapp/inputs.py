from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}


CancellationCheckpoint = Callable[[], None]
ProgressCallback = Callable[[int, str, str], None]


def resolve_frame_presentation_timestamps(
    decoded_timestamps_ms: list[float],
    source_frame_indexes: list[int],
    source_fps: float,
) -> tuple[list[float], str, float]:
    """Return bounded monotonic media times and an honest uncertainty label."""

    decoder_is_valid = (
        len(decoded_timestamps_ms) == len(source_frame_indexes)
        and bool(decoded_timestamps_ms)
        and all(math.isfinite(value) and value >= 0 for value in decoded_timestamps_ms)
        and all(
            decoded_timestamps_ms[index] > decoded_timestamps_ms[index - 1]
            for index in range(1, len(decoded_timestamps_ms))
        )
    )
    frame_period_ms = 1000.0 / max(1.0, source_fps)
    if decoder_is_valid:
        return (
            [round(float(value), 3) for value in decoded_timestamps_ms],
            "decoded-frame-pts",
            round(max(1.0, frame_period_ms / 2), 3),
        )
    return (
        [round(index * frame_period_ms, 3) for index in source_frame_indexes],
        "source-fps-fallback",
        round(frame_period_ms, 3),
    )


def validate_image_files(
    paths: list[Path],
    *,
    cancellation_checkpoint: CancellationCheckpoint | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 60_000_000
    for index, path in enumerate(paths):
        if cancellation_checkpoint is not None:
            cancellation_checkpoint()
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"invalid image upload: {path.name}") from exc
        if progress_callback is not None:
            progress_callback(
                12 + round(14 * (index + 1) / max(1, len(paths))),
                f"Validated image {index + 1} of {len(paths)}",
                "preprocessing",
            )


def extract_video_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    fps: int,
    max_frames: int,
    cancellation_checkpoint: CancellationCheckpoint | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    import cv2

    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("could not open the uploaded video")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    original_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = max(1, round(source_fps / max(fps, 1)))
    frame_index = 0
    saved: list[Path] = []
    decoded_timestamps_ms: list[float] = []
    source_frame_indexes: list[int] = []
    try:
        while len(saved) < max_frames:
            if cancellation_checkpoint is not None:
                cancellation_checkpoint()
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                destination = frames_dir / f"{len(saved):06d}.jpg"
                if not cv2.imwrite(str(destination), frame):
                    raise RuntimeError(f"could not write extracted frame {destination.name}")
                saved.append(destination)
                decoded_timestamps_ms.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)))
                source_frame_indexes.append(frame_index)
                if progress_callback is not None:
                    progress_callback(
                        min(28, 12 + round(16 * len(saved) / max_frames)),
                        f"Decoded {len(saved)} of at most {max_frames} frames",
                        "decoding",
                    )
            frame_index += 1
    finally:
        capture.release()

    if len(saved) < 2:
        raise ValueError("the video must yield at least two frames")
    frame_timestamps_ms, timestamp_association, timestamp_uncertainty_ms = (
        resolve_frame_presentation_timestamps(
            decoded_timestamps_ms,
            source_frame_indexes,
            source_fps,
        )
    )
    return saved, {
        "input_mode": "video",
        "source_fps": round(source_fps, 3),
        "sample_fps": fps,
        "sample_interval": interval,
        "original_frame_count": original_count,
        "frames_used": len(saved),
        "frame_presentation_timestamps_ms": frame_timestamps_ms,
        "frame_timestamp_association": timestamp_association,
        "frame_timestamp_uncertainty_ms": timestamp_uncertainty_ms,
    }


def stage_input_media(
    paths: list[Path],
    frames_dir: Path,
    *,
    fps: int,
    max_frames: int,
    summary_metadata: dict[str, Any] | None = None,
    cancellation_checkpoint: CancellationCheckpoint,
    progress_callback: ProgressCallback,
) -> dict[str, Any]:
    """Validate and stage one bounded provider input with cancellation checkpoints."""

    image_uploads = [path for path in paths if path.suffix.lower() in IMAGE_SUFFIXES]
    video_uploads = [path for path in paths if path.suffix.lower() in VIDEO_SUFFIXES]
    if video_uploads and image_uploads:
        raise ValueError("upload either one video or an image sequence")
    if len(video_uploads) > 1:
        raise ValueError("upload only one video")

    cancellation_checkpoint()
    if video_uploads:
        progress_callback(12, "Decoding the bounded video sequence", "decoding")
        _paths, summary = extract_video_frames(
            video_uploads[0],
            frames_dir,
            fps=fps,
            max_frames=max_frames,
            cancellation_checkpoint=cancellation_checkpoint,
            progress_callback=progress_callback,
        )
        cancellation_checkpoint()
        return {
            "image_dir": frames_dir,
            "input_summary": {**summary, **(summary_metadata or {})},
        }

    if len(image_uploads) < 2:
        raise ValueError("upload at least two images")
    selected = image_uploads[:max_frames]
    progress_callback(12, "Validating the bounded image sequence", "preprocessing")
    validate_image_files(
        selected,
        cancellation_checkpoint=cancellation_checkpoint,
        progress_callback=progress_callback,
    )
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(selected):
        cancellation_checkpoint()
        source.replace(frames_dir / f"{index:06d}{source.suffix.lower()}")
    cancellation_checkpoint()
    return {
        "image_dir": frames_dir,
        "input_summary": {
            "input_mode": "images",
            "frames_used": len(selected),
            **(summary_metadata or {}),
        },
    }
