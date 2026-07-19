import pytest
from PIL import Image

from core.exceptions import MediaLoadError
from media.image_file_loader import ImageFileLoader


def test_image_file_loader_loads_image(tmp_path):
    image_path = tmp_path / "id_card.jpg"

    Image.new(
        "RGB",
        (32, 32),
        color=(128, 128, 128),
    ).save(image_path)

    image = ImageFileLoader().load(image_path)

    assert image.mode == "RGB"
    assert image.size == (32, 32)


def test_image_file_loader_rejects_invalid_image(tmp_path):
    image_path = tmp_path / "bad.jpg"
    image_path.write_bytes(b"not an image")

    with pytest.raises(MediaLoadError):
        ImageFileLoader().load(image_path)
