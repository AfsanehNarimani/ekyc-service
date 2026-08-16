from abc import ABC, abstractmethod
from collections.abc import Sequence

from ai_modules.face_verification.evaluations.face_verification_metric_pair import FaceVerificationMetricPair


class BaseFaceVerificationMetricPairCalculator(ABC):
    @abstractmethod
    def calculate_at_threshold(
        self,
        genuine_similarity_scores: Sequence[float],
        imposter_similarity_scores: Sequence[float],
        threshold: float,
    )->FaceVerificationMetricPair:
        raise NotImplementedError
