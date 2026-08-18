import pytest

from ai_modules.face_verification.evaluations.face_verification_metric_curve_calculator import (
    FaceVerificationMetricCurveCalculator,
)
from ai_modules.face_verification.evaluations.face_verification_metric_pair_calculator import (
    FaceVerificationMetricPairCalculator,
)


def test_calculate_metric_operating_points_for_multiple_thresholds() -> None:
    metric_pair_calculator = FaceVerificationMetricPairCalculator()

    metric_curve_calculator = FaceVerificationMetricCurveCalculator(
        metric_pair_calculator=metric_pair_calculator,
    )

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
    thresholds = [
        0.40,
        0.50,
        0.80,
    ]

    operating_points = metric_curve_calculator.calculate_for_thresholds(
        genuine_similarity_scores=genuine_similarity_scores,
        impostor_similarity_scores=impostor_similarity_scores,
        thresholds=thresholds,
    )

    assert [
        operating_point.threshold for operating_point in operating_points
    ] == thresholds

    assert [
        operating_point.metric_pair.false_match_rate
        for operating_point in operating_points
    ] == pytest.approx(
        [
            0.75,
            0.50,
            0.25,
        ]
    )

    assert [
        operating_point.metric_pair.true_match_rate
        for operating_point in operating_points
    ] == pytest.approx(
        [
            1.00,
            0.75,
            0.25,
        ]
    )


def test_reject_empty_thresholds() -> None:
    metric_pair_calculator = FaceVerificationMetricPairCalculator()

    metric_curve_calculator = FaceVerificationMetricCurveCalculator(
        metric_pair_calculator=metric_pair_calculator,
    )

    with pytest.raises(
        ValueError,
        match=r"^Thresholds must not be empty\.$",
    ):
        metric_curve_calculator.calculate_for_thresholds(
            genuine_similarity_scores=[
                0.90,
                0.70,
            ],
            impostor_similarity_scores=[
                0.30,
                0.20,
            ],
            thresholds=[],
        )
