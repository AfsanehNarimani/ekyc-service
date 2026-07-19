from core.media_reference import MediaReference


def test_media_reference_detects_s3_uri():
    reference = MediaReference("s3://my-bucket/path/video.mp4")

    assert reference.is_s3_uri() is True
    assert reference.is_http_uri() is False
    assert reference.is_local_path() is False


def test_media_reference_detects_http_uri():
    reference = MediaReference("https://example.com/video.mp4")

    assert reference.is_s3_uri() is False
    assert reference.is_http_uri() is True
    assert reference.is_local_path() is False


def test_media_reference_detects_local_path():
    reference = MediaReference("/tmp/video.mp4")

    assert reference.is_s3_uri() is False
    assert reference.is_http_uri() is False
    assert reference.is_local_path() is True
