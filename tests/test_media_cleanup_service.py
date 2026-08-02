from config import settings
from media.media_cleanup_service import MediaCleanupService


def test_media_cleanup_service_deletes_file_inside_temp_dir(tmp_path, monkeypatch):
    temp_root = tmp_path / "media"
    temp_root.mkdir()

    monkeypatch.setattr(
        settings,
        "media_temp_dir",
        temp_root,
    )

    temp_file = temp_root / "downloaded.mp4"
    temp_file.write_text("video")

    MediaCleanupService().cleanup_temp_file(
        path=temp_file,
        correlation_id="trace-1",
    )

    assert temp_file.exists() is False


def test_media_cleanup_service_does_not_delete_file_outside_temp_dir(
    tmp_path,
    monkeypatch,
):
    temp_root = tmp_path / "media"
    temp_root.mkdir()

    local_file = tmp_path / "user_video.mp4"
    local_file.write_text("video")

    monkeypatch.setattr(
        settings,
        "media_temp_dir",
        temp_root,
    )

    MediaCleanupService().cleanup_temp_file(
        path=local_file,
        correlation_id="trace-1",
    )

    assert local_file.exists() is True
