import pytest

from ai_modules.face_verification.evaluations.face_verification_metric_pair_calculator import (
    FaceVerificationMetricPairCalculator,
)


def test_calculate_fmr_and_fnmr_at_threshold() -> None:
    calculator = FaceVerificationMetricPairCalculator()

    genuine_similarity_scores = [
        0.90,
        0.70,
        0.50,
        0.49,
    ]
    imposter_similarity_scores = [
        0.80,
        0.50,
        0.49,
        0.20,
    ]

    result = calculator.calculate_at_threshold(
        genuine_similarity_scores=genuine_similarity_scores,
        imposter_similarity_scores=imposter_similarity_scores,
        threshold=0.50,
    )

    assert result.false_match_rate == pytest.approx(0.50)
    assert result.false_non_match_rate == pytest.approx(0.25)