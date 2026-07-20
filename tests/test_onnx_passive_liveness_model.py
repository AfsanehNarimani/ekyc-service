import numpy as np
import pytest

from liveness.onnx_passive_liveness_model import ONNXPassiveLivenessModel


def test_onnx_passive_liveness_model_extracts_real_score_from_configured_index():
    model = ONNXPassiveLivenessModel.__new__(ONNXPassiveLivenessModel)
    model.real_class_index = 1

    score = model._extract_real_score(
        np.array(
            [
                0.0,
                2.0,
            ],
            dtype=np.float32,
        )
    )

    assert score > 0.8


def test_onnx_passive_liveness_model_rejects_invalid_real_class_index():
    model = ONNXPassiveLivenessModel.__new__(ONNXPassiveLivenessModel)
    model.real_class_index = 3

    with pytest.raises(ValueError):
        model._extract_real_score(
            np.array(
                [
                    0.0,
                    2.0,
                ],
                dtype=np.float32,
            )
        )
