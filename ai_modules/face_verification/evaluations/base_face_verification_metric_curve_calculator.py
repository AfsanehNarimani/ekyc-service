from abc import ABC, abstractmethod
from typing import Sequence

from ai_modules.face_verification.evaluations.face_verification_metric_operating_point import \
    FaceVerificationMetricOperatingPoint


class BaseFaceVerificationMetricCurveCalculator(ABC):
    @abstractmethod
    def calculate_for_thresholds(
        self,
        genuine_similarity_scores:Sequence[float],
        impostor_similarity_scores:Sequence[float],
        thresholds: Sequence[float]) -> tuple[FaceVerificationMetricOperatingPoint, ...]:
            raise NotImplementedError