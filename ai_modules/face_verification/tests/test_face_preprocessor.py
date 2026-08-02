import numpy as np
from PIL import Image

from core.face_detection import FaceDetection
from ai_modules.face_verification.infrastructure.preprocessing.face_preprocessor import FacePreprocessor


class FakeDetector:
    def __init__(
        self,
        detection: FaceDetection,
    ) -> None:
        self.detection = detection
        self.call_count = 0

    def detect(
        self,
        image: Image.Image,
    ) -> list[FaceDetection]:
        self.call_count += 1
        return [self.detection]


class FakeValidator:
    def __init__(self) -> None:
        self.call_count = 0

    def validate(
        self,
        detections: list[FaceDetection],
    ) -> FaceDetection:
        self.call_count += 1
        return detections[0]


class FakeAligner:
    def __init__(self) -> None:
        self.call_count = 0
        self.received_detection = None

    def align(
        self,
        image: Image.Image,
        detection: FaceDetection,
    ) -> Image.Image:
        self.call_count += 1
        self.received_detection = detection
        return Image.new(
            "RGB",
            (
                112,
                112,
            ),
        )


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
    )


def test_face_preprocessor_detects_validates_and_aligns():
    detection = make_detection()
    detector = FakeDetector(detection)
    validator = FakeValidator()
    aligner = FakeAligner()

    preprocessor = FacePreprocessor(
        detector=detector,
        validator=validator,
        aligner=aligner,
    )

    result = preprocessor.preprocess(
        Image.new(
            "RGB",
            (
                200,
                200,
            ),
        )
    )

    assert result.size == (
        112,
        112,
    )
    assert detector.call_count == 1
    assert validator.call_count == 1
    assert aligner.call_count == 1
    assert aligner.received_detection is detection


def test_face_preprocessor_reuses_precomputed_detection():
    detection = make_detection()
    detector = FakeDetector(detection)
    validator = FakeValidator()
    aligner = FakeAligner()

    preprocessor = FacePreprocessor(
        detector=detector,
        validator=validator,
        aligner=aligner,
    )

    result = preprocessor.preprocess_with_detection(
        image=Image.new(
            "RGB",
            (
                200,
                200,
            ),
        ),
        detection=detection,
    )

    assert result.size == (
        112,
        112,
    )
    assert detector.call_count == 0
    assert validator.call_count == 0
    assert aligner.call_count == 1
    assert aligner.received_detection is detection
