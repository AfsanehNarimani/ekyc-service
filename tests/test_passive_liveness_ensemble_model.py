import numpy as np
import pytest

from liveness.passive_liveness_ensemble_model import (
    PassiveLivenessEnsembleModel,
)


class FakeProbabilityModel:
    def __init__(
        self,
        probabilities: np.ndarray,
    ) -> None:
        self.probabilities = probabilities

    def predict_probabilities(
        self,
        frame_bgr,
        detection,
    ) -> np.ndarray:
        return self.probabilities


def test_passive_liveness_ensemble_averages_model_probabilities():
    model = PassiveLivenessEnsembleModel(
        models=[
            FakeProbabilityModel(
                np.array(
                    [
                        0.10,
                        0.80,
                        0.10,
                    ],
                    dtype=np.float32,
                )
            ),
            FakeProbabilityModel(
                np.array(
                    [
                        0.20,
                        0.60,
                        0.20,
                    ],
                    dtype=np.float32,
                )
            ),
        ],
        real_class_index=1,
    )

    score = model.predict_real_score(
        frame_bgr=np.zeros(
            (
                80,
                80,
                3,
            ),
            dtype=np.uint8,
        ),
        detection=None,
    )

    assert score == pytest.approx(0.70)


def test_passive_liveness_ensemble_rejects_empty_models():
    with pytest.raises(ValueError):
        PassiveLivenessEnsembleModel(models=[])


def test_passive_liveness_ensemble_rejects_incompatible_outputs():
    model = PassiveLivenessEnsembleModel(
        models=[
            FakeProbabilityModel(
                np.array(
                    [
                        0.10,
                        0.80,
                        0.10,
                    ],
                    dtype=np.float32,
                )
            ),
            FakeProbabilityModel(
                np.array(
                    [
                        0.20,
                        0.80,
                    ],
                    dtype=np.float32,
                )
            ),
        ]
    )

    with pytest.raises(ValueError):
        model.predict_probabilities(
            frame_bgr=np.zeros(
                (
                    80,
                    80,
                    3,
                ),
                dtype=np.uint8,
            ),
            detection=None,
        )


def test_passive_liveness_ensemble_rejects_invalid_real_class_index():
    model = PassiveLivenessEnsembleModel(
        models=[
            FakeProbabilityModel(
                np.array(
                    [
                        0.10,
                        0.80,
                        0.10,
                    ],
                    dtype=np.float32,
                )
            )
        ],
        real_class_index=3,
    )

    with pytest.raises(ValueError):
        model.predict_real_score(
            frame_bgr=np.zeros(
                (
                    80,
                    80,
                    3,
                ),
                dtype=np.uint8,
            ),
            detection=None,
        )
