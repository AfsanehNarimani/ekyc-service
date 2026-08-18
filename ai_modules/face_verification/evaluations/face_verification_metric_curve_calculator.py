from typing import Sequence

from ai_modules.face_verification.evaluations.base_face_verification_metric_curve_calculator import (
    BaseFaceVerificationMetricCurveCalculator,
)
from ai_modules.face_verification.evaluations.base_face_verification_metric_pair_calculator import (
    BaseFaceVerificationMetricPairCalculator,
)
from ai_modules.face_verification.evaluations.face_verification_metric_operating_point import (
    FaceVerificationMetricOperatingPoint,
)


class FaceVerificationMetricCurveCalculator(BaseFaceVerificationMetricCurveCalculator):
    def __init__(
        self,
        metric_pair_calculator: BaseFaceVerificationMetricPairCalculator,
    ) -> None:
        self.metric_pair_calculator = metric_pair_calculator

    def calculate_for_thresholds(
        self,
        genuine_similarity_scores: Sequence[float],
        impostor_similarity_scores: Sequence[float],
        thresholds: Sequence[float],
    ) -> tuple[FaceVerificationMetricOperatingPoint, ...]:
        if not thresholds:
            raise ValueError("Thresholds must not be empty.")
        operation_points = []

        for threshold in thresholds:
            metric_pair = self.metric_pair_calculator.calculate_at_threshold(
                genuine_similarity_scores=genuine_similarity_scores,
                impostor_similarity_scores=impostor_similarity_scores,
                threshold=threshold,
            )

            operation_points.append(
                FaceVerificationMetricOperatingPoint(
                    threshold=threshold,
                    metric_pair=metric_pair,
                )
            )

        return tuple(operation_points)
