import numpy as np
import pytest

from ai_modules.face_detection.domain.face_detection import FaceDetection
from ai_modules.face_detection.domain.single_face_validator import SingleFaceValidator
from core.exceptions import MultipleFacesDetectedError, NoFaceDetectedError


def make_detection() -> FaceDetection:
    return FaceDetection(
        bbox=(0.0, 0.0, 100.0, 100.0),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        score=0.99,
    )


def test_single_face_validator_accepts_one_face():
    detection = make_detection()

    result = SingleFaceValidator().validate([detection])

    assert result is detection


def test_single_face_validator_rejects_no_face():
    with pytest.raises(NoFaceDetectedError):
        SingleFaceValidator().validate([])


def test_single_face_validator_rejects_multiple_faces():
    detection = make_detection()

    with pytest.raises(MultipleFacesDetectedError):
        SingleFaceValidator().validate([detection, detection])
