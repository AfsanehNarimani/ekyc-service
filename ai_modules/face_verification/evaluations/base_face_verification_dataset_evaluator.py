from abc import ABC, abstractmethod
from collections.abc import Iterator

from ai_modules.face_verification.evaluations.base_face_verification_evaluation_dataset import (
    BaseFaceVerificationEvaluationDataset,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_evaluation_result import (
    FaceVerificationComparisonEvaluationResult,
)


class BaseFaceVerificationDatasetEvaluator(ABC):
    @abstractmethod
    def iter_comparison_evaluation_results(
        self,
        evaluation_dataset: BaseFaceVerificationEvaluationDataset,
    ) -> Iterator[FaceVerificationComparisonEvaluationResult]:
        raise NotImplementedError()
