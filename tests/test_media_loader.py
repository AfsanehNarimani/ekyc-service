import base64

import pytest

from core.exceptions import MediaLoadError
from core.media_reference import MediaReference
from media.media_loader import MediaLoader


def test_media_loader_returns_existing_local_file(tmp_path):
    file_path = tmp_path / "sample.mp4"
    file_path.write_bytes(b"fake video")

    loader = MediaLoader(
        temp_dir=tmp_path / "tmp",
    )

    result = loader.load_to_file(
        reference=MediaReference(str(file_path)),
        suffix=".mp4",
    )

    assert result == file_path


def test_media_loader_decodes_base64_data_uri(tmp_path):
    payload = b"simulated-person-image"
    encoded = base64.b64encode(payload).decode("ascii")
    uri = f"data:application/octet-stream;base64,{encoded}"

    loader = MediaLoader(
        temp_dir=tmp_path / "tmp",
    )

    result = loader.load_to_file(
        reference=MediaReference(uri),
        suffix=".jpg",
    )

    assert result.read_bytes() == payload
    assert result.suffix == ".jpg"


def test_media_loader_rejects_malformed_base64_data_uri(tmp_path):
    uri = "data:application/octet-stream;base64,not-valid-base64!!"

    loader = MediaLoader(
        temp_dir=tmp_path / "tmp",
    )

    with pytest.raises(MediaLoadError):
        loader.load_to_file(
            reference=MediaReference(uri),
            suffix=".jpg",
        )


def test_media_loader_rejects_non_base64_data_uri(tmp_path):
    uri = "data:text/plain,hello"

    loader = MediaLoader(
        temp_dir=tmp_path / "tmp",
    )

    with pytest.raises(MediaLoadError):
        loader.load_to_file(
            reference=MediaReference(uri),
            suffix=".jpg",
        )
