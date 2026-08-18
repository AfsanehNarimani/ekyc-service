from collections.abc import Iterator
from pathlib import Path

from ai_modules.face_verification.evaluations.base_face_verification_evaluation_dataset import (
    BaseFaceVerificationEvaluationDataset,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_evaluation_result import (
    FaceVerificationComparisonEvaluationResult,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_sample import (
    FaceVerificationComparisonSample,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_status import (
    FaceVerificationComparisonStatus,
)
from ai_modules.face_verification.evaluations.face_verification_dataset_evaluator import (
    FaceVerificationDatasetEvaluator,
)


class FakeFaceVerificationEvaluationDataset(BaseFaceVerificationEvaluationDataset):
    def __init__(
        self,
        comparison_samples: list[FaceVerificationComparisonSample],
    ) -> None:
        self.comparison_samples = comparison_samples

    @property
    def dataset_name(self) -> str:
        return "internal-ekyc"

    def iter_comparison_samples(
        self,
    ) -> Iterator[FaceVerificationComparisonSample]:
        yield from self.comparison_samples


class FakeFaceVerificationComparisonEvaluator:
    def __init__(self) -> None:
        self.evaluated_comparison_ids: list[str] = []
        self.received_dataset_names: list[str] = []

    def evaluate_comparison(
        self,
        dataset_name: str,
        comparison_sample: FaceVerificationComparisonSample,
    ) -> FaceVerificationComparisonEvaluationResult:
        self.received_dataset_names.append(dataset_name)
        self.evaluated_comparison_ids.append(comparison_sample.comparison_id)

        return FaceVerificationComparisonEvaluationResult(
            dataset_name=dataset_name,
            comparison_sample=comparison_sample,
            model_name="edgeface",
            status=FaceVerificationComparisonStatus.SUCCESS,
            processing_time_milliseconds=1.0,
            similarity_score=0.8,
        )


def create_comparison_sample(
    comparison_id: str,
) -> FaceVerificationComparisonSample:
    return FaceVerificationComparisonSample(
        comparison_id=comparison_id,
        reference_subject_id="subject-001",
        candidate_subject_id="subject-001",
        reference_image_path=Path(f"{comparison_id}-reference.jpg"),
        candidate_image_path=Path(f"{comparison_id}-candidate.jpg"),
        is_genuine=True,
    )


def test_iterate_dataset_comparison_evaluation_results_lazily() -> None:
    comparison_evaluator = FakeFaceVerificationComparisonEvaluator()
    evaluation_dataset = FakeFaceVerificationEvaluationDataset(
        comparison_samples=[
            create_comparison_sample("comparison-001"),
            create_comparison_sample("comparison-002"),
        ],
    )
    dataset_evaluator = FaceVerificationDatasetEvaluator(
        comparison_evaluator=comparison_evaluator,
    )

    evaluation_results = dataset_evaluator.iter_comparison_evaluation_results(
        evaluation_dataset=evaluation_dataset,
    )

    assert comparison_evaluator.evaluated_comparison_ids == []

    first_result = next(evaluation_results)

    assert first_result.comparison_sample.comparison_id == ("comparison-001")
    assert comparison_evaluator.evaluated_comparison_ids == ["comparison-001"]

    remaining_results = list(evaluation_results)

    assert [result.comparison_sample.comparison_id for result in remaining_results] == [
        "comparison-002"
    ]
    assert comparison_evaluator.evaluated_comparison_ids == [
        "comparison-001",
        "comparison-002",
    ]
    assert comparison_evaluator.received_dataset_names == [
        "internal-ekyc",
        "internal-ekyc",
    ]
