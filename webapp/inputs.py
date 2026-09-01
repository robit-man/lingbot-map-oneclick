from __future__ import annotations

from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}


def validate_image_files(paths: list[Path]) -> None:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 60_000_000
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"invalid image upload: {path.name}") from exc


def extract_video_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    fps: int,
    max_frames: int,
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
    try:
        while len(saved) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                destination = frames_dir / f"{len(saved):06d}.jpg"
                if not cv2.imwrite(str(destination), frame):
                    raise RuntimeError(f"could not write extracted frame {destination.name}")
                saved.append(destination)
            frame_index += 1
    finally:
        capture.release()

    if len(saved) < 2:
        raise ValueError("the video must yield at least two frames")
    return saved, {
        "input_mode": "video",
        "source_fps": round(source_fps, 3),
        "sample_fps": fps,
        "sample_interval": interval,
        "original_frame_count": original_count,
        "frames_used": len(saved),
    }
