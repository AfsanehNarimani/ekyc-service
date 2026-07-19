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
