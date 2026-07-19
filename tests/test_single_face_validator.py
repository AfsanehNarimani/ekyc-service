import numpy as np
import pytest

from core.exceptions import MultipleFacesDetectedError, NoFaceDetectedError
from core.face_detection import FaceDetection
from face.single_face_validator import SingleFaceValidator


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
