from pathlib import Path

import pytest
from PIL import Image

from webapp.inputs import validate_image_files


def test_validate_image_files_accepts_decodable_images(tmp_path: Path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (32, 24), (10, 20, 30)).save(image_path)
    validate_image_files([image_path])


def test_validate_image_files_rejects_invalid_payload(tmp_path: Path):
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="invalid image upload"):
        validate_image_files([image_path])
