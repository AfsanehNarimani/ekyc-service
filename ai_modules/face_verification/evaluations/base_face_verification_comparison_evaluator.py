from abc import ABC, abstractmethod

from ai_modules.face_verification.evaluations.face_verification_comparison_evaluation_result import (
    FaceVerificationComparisonEvaluationResult,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_sample import (
    FaceVerificationComparisonSample,
)


class BaseFaceVerificationComparisonEvaluator(ABC):
    @abstractmethod
    def evaluate_comparison(
        self, dataset_name: str, comparison_sample: FaceVerificationComparisonSample
    ) -> FaceVerificationComparisonEvaluationResult:
        raise NotImplementedError
