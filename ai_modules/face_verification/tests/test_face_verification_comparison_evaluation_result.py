from pathlib import Path

import pytest

from ai_modules.face_verification.evaluations.face_verification_comparison_evaluation_result import (
    FaceVerificationComparisonEvaluationResult,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_sample import (
    FaceVerificationComparisonSample,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_status import (
    FaceVerificationComparisonStatus,
)


def create_comparison_sample() -> FaceVerificationComparisonSample:
    return FaceVerificationComparisonSample(
        comparison_id="comparison-001",
        reference_subject_id="subject-001",
        candidate_subject_id="subject-001",
        reference_image_path=Path("reference.jpg"),
        candidate_image_path=Path("candidate.jpg"),
        is_genuine=True,
    )


def test_create_successful_comparison_result() -> None:
    result = FaceVerificationComparisonEvaluationResult(
        dataset_name="internal-ekyc",
        comparison_sample=create_comparison_sample(),
        model_name="edgeface",
        status=FaceVerificationComparisonStatus.SUCCESS,
        processing_time_milliseconds=42.7,
        similarity_score=0.82,
    )

    assert result.status is FaceVerificationComparisonStatus.SUCCESS
    assert result.similarity_score == pytest.approx(0.82)
    assert result.failure_detail is None


def test_create_failed_comparison_result() -> None:
    result = FaceVerificationComparisonEvaluationResult(
        dataset_name="internal-ekyc",
        comparison_sample=create_comparison_sample(),
        model_name="edgeface",
        status=(FaceVerificationComparisonStatus.CANDIDATE_FACE_NOT_DETECTED),
        processing_time_milliseconds=18.4,
        failure_detail=("No face was detected in the candidate image."),
    )

    assert result.status is (
        FaceVerificationComparisonStatus.CANDIDATE_FACE_NOT_DETECTED
    )
    assert result.similarity_score is None


def test_reject_successful_result_without_similarity_score() -> None:
    with pytest.raises(
        ValueError,
        match=(r"^Successful comparison must have " r"a similarity score\.$"),
    ):
        FaceVerificationComparisonEvaluationResult(
            dataset_name="internal-ekyc",
            comparison_sample=create_comparison_sample(),
            model_name="edgeface",
            status=FaceVerificationComparisonStatus.SUCCESS,
            processing_time_milliseconds=42.7,
        )


def test_reject_failed_result_with_similarity_score() -> None:
    with pytest.raises(
        ValueError,
        match=(r"^Failed comparison must not have " r"a similarity score\.$"),
    ):
        FaceVerificationComparisonEvaluationResult(
            dataset_name="internal-ekyc",
            comparison_sample=create_comparison_sample(),
            model_name="edgeface",
            status=(FaceVerificationComparisonStatus.CANDIDATE_FACE_NOT_DETECTED),
            processing_time_milliseconds=18.4,
            similarity_score=0.82,
        )


def test_reject_negative_processing_time() -> None:
    with pytest.raises(
        ValueError,
        match=r"^Processing time must not be negative\.$",
    ):
        FaceVerificationComparisonEvaluationResult(
            dataset_name="internal-ekyc",
            comparison_sample=create_comparison_sample(),
            model_name="edgeface",
            status=FaceVerificationComparisonStatus.SUCCESS,
            processing_time_milliseconds=-1.0,
            similarity_score=0.82,
        )


@pytest.mark.parametrize(
    "processing_time_milliseconds",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_reject_non_finite_processing_time(
    processing_time_milliseconds: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"^Processing time must be finite\.$",
    ):
        FaceVerificationComparisonEvaluationResult(
            dataset_name="internal-ekyc",
            comparison_sample=create_comparison_sample(),
            model_name="edgeface",
            status=FaceVerificationComparisonStatus.SUCCESS,
            processing_time_milliseconds=processing_time_milliseconds,
            similarity_score=0.82,
        )
