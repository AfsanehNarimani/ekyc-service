import numpy as np
from PIL import Image

from ai_modules.face_detection.application.face_detection_service import (
    FaceDetectionService,
)
from ai_modules.face_detection.domain.face_detection import FaceDetection
from ai_modules.face_detection.domain.single_face_validator import (
    SingleFaceValidator,
)
from core.video_frame import VideoFrame


class FakeDetector:
    def __init__(
        self,
        results: list[list[FaceDetection]],
    ) -> None:
        self.results = results
        self.call_count = 0

    def detect(
        self,
        image: Image.Image,
    ) -> list[FaceDetection]:
        result = self.results[self.call_count]
        self.call_count += 1
        return result


def make_detection() -> FaceDetection:
    return FaceDetection(
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
        pose=(
            0.0,
            0.0,
            0.0,
        ),
    )


def make_frame(
    frame_index: int,
) -> VideoFrame:
    return VideoFrame(
        frame_index=frame_index,
        timestamp_seconds=float(frame_index),
        image=Image.new(
            "RGB",
            (
                100,
                100,
            ),
        ),
    )


def test_detect_from_image_returns_single_face():
    detection = make_detection()
    detector = FakeDetector(
        results=[
            [detection],
        ]
    )

    service = FaceDetectionService(
        detector=detector,
        validator=SingleFaceValidator(),
    )

    result = service.detect_from_image(
        image=Image.new(
            "RGB",
            (
                100,
                100,
            ),
        ),
    )

    assert result is detection
    assert detector.call_count == 1


def test_detect_from_frames_detects_each_frame_once():
    detection = make_detection()
    frames = [
        make_frame(0),
        make_frame(1),
    ]
    detector = FakeDetector(
        results=[
            [detection],
            [detection],
        ]
    )

    service = FaceDetectionService(
        detector=detector,
        validator=SingleFaceValidator(),
    )

    result = service.detect_from_frames(
        frames=frames,
    )

    assert detector.call_count == 2
    assert result.total_frames == 2
    assert result.skipped_frames == 0
    assert len(result.frames) == 2
    assert result.frames[0].frame is frames[0]
    assert result.frames[0].detection is detection


def test_detect_from_frames_skips_invalid_frames():
    detection = make_detection()
    frames = [
        make_frame(0),
        make_frame(1),
        make_frame(2),
    ]
    detector = FakeDetector(
        results=[
            [detection],
            [],
            [
                detection,
                detection,
            ],
        ]
    )

    service = FaceDetectionService(
        detector=detector,
        validator=SingleFaceValidator(),
    )

    result = service.detect_from_frames(
        frames=frames,
    )

    assert detector.call_count == 3
    assert result.total_frames == 3
    assert result.skipped_frames == 2
    assert len(result.frames) == 1
    assert result.frames[0].frame is frames[0]
