import numpy as np
import pytest

from core.face_detection import FaceDetection
from liveness.passive_liveness_preprocessor import (
    PassiveLivenessPreprocessor,
)


def test_passive_liveness_preprocessor_returns_nchw_tensor():
    frame = np.zeros(
        (
            200,
            200,
            3,
        ),
        dtype=np.uint8,
    )

    detection = FaceDetection(
        bbox=(
            50,
            50,
            150,
            150,
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

    result = PassiveLivenessPreprocessor(
        input_width=80,
        input_height=80,
        crop_scale=2.7,
    ).preprocess(
        frame_bgr=frame,
        detection=detection,
    )

    assert result.tensor.shape == (
        1,
        3,
        80,
        80,
    )
    assert result.tensor.dtype == np.float32


def test_passive_liveness_preprocessor_rejects_invalid_crop_scale():
    with pytest.raises(ValueError):
        PassiveLivenessPreprocessor(
            input_width=80,
            input_height=80,
            crop_scale=1.0,
        )


def test_passive_liveness_preprocessor_preserves_raw_pixel_scale():
    frame = np.full(
        (
            200,
            200,
            3,
        ),
        fill_value=255,
        dtype=np.uint8,
    )

    detection = FaceDetection(
        bbox=(
            50,
            50,
            150,
            150,
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

    result = PassiveLivenessPreprocessor(
        input_width=80,
        input_height=80,
        crop_scale=2.7,
        pixel_divisor=1.0,
    ).preprocess(
        frame_bgr=frame,
        detection=detection,
    )

    assert np.max(result.tensor) == pytest.approx(255.0)


def test_passive_liveness_preprocessor_rejects_invalid_pixel_divisor():
    with pytest.raises(ValueError):
        PassiveLivenessPreprocessor(
            input_width=80,
            input_height=80,
            crop_scale=2.7,
            pixel_divisor=0.0,
        )
