from pathlib import Path

import cv2
import numpy as np
import pytest

from core.exceptions import InvalidVideoError
from video_processing.video_frame_extractor import VideoFrameExtractor


def create_test_video(
    path: Path,
    fps: int = 10,
    total_frames: int = 30,
) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (64, 64),
    )

    for index in range(total_frames):
        frame = np.full(
            (64, 64, 3),
            fill_value=index,
            dtype=np.uint8,
        )
        writer.write(frame)

    writer.release()


def test_video_frame_extractor_extracts_frames_by_interval(tmp_path):
    video_path = tmp_path / "test.avi"
    create_test_video(video_path)

    extractor = VideoFrameExtractor(
        frame_interval_seconds=1.0,
        max_frames=3,
    )

    frames = extractor.extract(video_path)

    assert len(frames) == 3
    assert frames[0].frame_index == 0
    assert frames[1].frame_index == 10
    assert frames[2].frame_index == 20
    assert frames[0].image.size == (64, 64)
    assert frames[0].image.mode == "RGB"


def test_video_frame_extractor_rejects_missing_video(tmp_path):
    extractor = VideoFrameExtractor()

    with pytest.raises(FileNotFoundError):
        extractor.extract(tmp_path / "missing.mp4")


def test_video_frame_extractor_rejects_invalid_video(tmp_path):
    video_path = tmp_path / "bad.mp4"
    video_path.write_bytes(b"not a video")

    extractor = VideoFrameExtractor()

    with pytest.raises(InvalidVideoError):
        extractor.extract(video_path)
