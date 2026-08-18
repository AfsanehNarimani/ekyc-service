import math
from dataclasses import dataclass

from ai_modules.face_verification.evaluations.face_verification_comparison_sample import (
    FaceVerificationComparisonSample,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_status import (
    FaceVerificationComparisonStatus,
)


@dataclass(frozen=True, slots=True)
class FaceVerificationComparisonEvaluationResult:
    dataset_name: str
    comparison_sample: FaceVerificationComparisonSample
    model_name: str
    status: FaceVerificationComparisonStatus
    processing_time_milliseconds: float
    similarity_score: float | None = None
    failure_detail: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.processing_time_milliseconds):
            raise ValueError("Processing time must be finite.")

        if self.processing_time_milliseconds < 0:
            raise ValueError("Processing time must not be negative.")

        if (
            self.status is FaceVerificationComparisonStatus.SUCCESS
            and self.similarity_score is None
        ):
            raise ValueError("Successful comparison must have a similarity score.")

        if (
            self.status is not FaceVerificationComparisonStatus.SUCCESS
            and self.similarity_score is not None
        ):
            raise ValueError(
                "Failed comparison must not have a similarity score.",
            )

        if self.similarity_score is not None and not math.isfinite(
            self.similarity_score
        ):
            raise ValueError(
                "Similarity score must be finite.",
            )
