from pathlib import Path

import numpy as np
from PIL import Image

from core.face_detection import FaceDetection
from core.video_frame import VideoFrame
from services.video_frame_analysis_service import (
    VideoFrameAnalysisService,
)


class FakeFrameExtractor:
    def __init__(
        self,
        frames: list[VideoFrame],
    ) -> None:
        self.frames = frames
        self.call_count = 0

    def extract(
        self,
        video_path: Path,
    ) -> list[VideoFrame]:
        self.call_count += 1
        return self.frames


class FakeDetector:
    def __init__(
        self,
        detection: FaceDetection,
    ) -> None:
        self.detection = detection
        self.call_count = 0

    def detect(
        self,
        image,
    ) -> list[FaceDetection]:
        self.call_count += 1
        return [self.detection]


class FakeValidator:
    def validate(
        self,
        detections: list[FaceDetection],
    ) -> FaceDetection:
        return detections[0]


def test_video_frame_analysis_extracts_and_detects_each_frame_once():
    image = Image.new(
        "RGB",
        (
            100,
            100,
        ),
    )

    frames = [
        VideoFrame(
            frame_index=0,
            timestamp_seconds=0.0,
            image=image,
        ),
        VideoFrame(
            frame_index=25,
            timestamp_seconds=1.0,
            image=image,
        ),
    ]

    detection = FaceDetection(
        bbox=(
            10.0,
            10.0,
            90.0,
            90.0,
        ),
        landmarks=np.zeros(
            (
                5,
                2,
            ),
            dtype=np.float32,
        ),
        score=0.99,
    )

    frame_extractor = FakeFrameExtractor(frames)
    detector = FakeDetector(detection)

    service = VideoFrameAnalysisService(
        frame_extractor=frame_extractor,
        detector=detector,
        validator=FakeValidator(),
    )

    result = service.analyze(
        video_path=Path("/tmp/video.mp4"),
    )

    assert frame_extractor.call_count == 1
    assert detector.call_count == 2
    assert result.total_frames == 2
    assert result.skipped_frames == 0
    assert len(result.frames) == 2
    assert result.frames[0].frame.frame_index == 0
    assert result.frames[0].detection is detection
