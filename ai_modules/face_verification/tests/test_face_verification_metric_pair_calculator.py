import pytest
import math

from ai_modules.face_verification.evaluations.face_verification_metric_pair_calculator import (
    FaceVerificationMetricPairCalculator,
)


def test_calculate_fmr_and_tmr_at_threshold() -> None:
    calculator = FaceVerificationMetricPairCalculator()

    genuine_similarity_scores = [
        0.90,
        0.70,
        0.50,
        0.49,
    ]
    impostor_similarity_scores = [
        0.80,
        0.50,
        0.49,
        0.20,
    ]

    result = calculator.calculate_at_threshold(
        genuine_similarity_scores=genuine_similarity_scores,
        impostor_similarity_scores=impostor_similarity_scores,
        threshold=0.50,
    )

    assert result.false_match_rate == pytest.approx(0.50)
    assert result.true_match_rate == pytest.approx(0.75)


def test_reject_empty_genuine_similarity_scores() -> None:
    calculator = FaceVerificationMetricPairCalculator()

    with pytest.raises(
        ValueError,
        match=r"^Genuine similarity scores must not be empty\.$",
    ):
        calculator.calculate_at_threshold(
            genuine_similarity_scores=[],
            impostor_similarity_scores=[
                0.80,
                0.50,
                0.49,
                0.20,
            ],
            threshold=0.50,
        )


def test_reject_empty_impostor_similarity_scores() -> None:
    calculator = FaceVerificationMetricPairCalculator()

    with pytest.raises(
        ValueError,
        match=r"^Impostor similarity scores must not be empty\.$",
    ):
        calculator.calculate_at_threshold(
            genuine_similarity_scores=[
                0.90,
                0.70,
                0.50,
                0.49,
            ],
            impostor_similarity_scores=[],
            threshold=0.50,
        )
@pytest.mark.parametrize(
    "invalid_similarity_score",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_reject_non_finite_similarity_scores(
    invalid_similarity_score: float,
) -> None:
    calculator = FaceVerificationMetricPairCalculator()

    with pytest.raises(
        ValueError,
        match=(
            r"^Similarity scores must contain "
            r"only finite values\.$"
        ),
    ):
        calculator.calculate_at_threshold(
            genuine_similarity_scores=[
                0.90,
                invalid_similarity_score,
            ],
            impostor_similarity_scores=[
                0.30,
                0.20,
            ],
            threshold=0.50,
        )

@pytest.mark.parametrize(
    "invalid_threshold",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_reject_non_finite_threshold(
    invalid_threshold: float,
) -> None:
    calculator = FaceVerificationMetricPairCalculator()

    with pytest.raises(
        ValueError,
        match=r"^Threshold must be finite\.$",
    ):
        calculator.calculate_at_threshold(
            genuine_similarity_scores=[
                0.90,
                0.70,
            ],
            impostor_similarity_scores=[
                0.30,
                0.20,
            ],
            threshold=invalid_threshold,
        )