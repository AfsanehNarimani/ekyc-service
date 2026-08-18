from collections.abc import Iterator

from ai_modules.face_verification.evaluations.base_face_verification_comparison_evaluator import (
    BaseFaceVerificationComparisonEvaluator,
)
from ai_modules.face_verification.evaluations.base_face_verification_dataset_evaluator import (
    BaseFaceVerificationDatasetEvaluator,
)
from ai_modules.face_verification.evaluations.base_face_verification_evaluation_dataset import (
    BaseFaceVerificationEvaluationDataset,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_evaluation_result import (
    FaceVerificationComparisonEvaluationResult,
)


class FaceVerificationDatasetEvaluator(BaseFaceVerificationDatasetEvaluator):
    def __init__(
        self,
        comparison_evaluator: BaseFaceVerificationComparisonEvaluator,
    ) -> None:
        self.comparison_evaluator = comparison_evaluator

    def iter_comparison_evaluation_results(
        self,
        evaluation_dataset: BaseFaceVerificationEvaluationDataset,
    ) -> Iterator[FaceVerificationComparisonEvaluationResult]:
        for comparison_sample in evaluation_dataset.iter_comparison_samples():
            yield self.comparison_evaluator.evaluate_comparison(
                dataset_name=evaluation_dataset.dataset_name,
                comparison_sample=comparison_sample,
            )
