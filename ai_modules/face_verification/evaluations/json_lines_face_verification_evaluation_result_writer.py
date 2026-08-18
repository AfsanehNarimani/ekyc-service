import json
from pathlib import Path
from typing import TextIO

from ai_modules.face_verification.evaluations.base_face_verification_evaluation_result_writer import (
    BaseFaceVerificationEvaluationResultWriter,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_evaluation_result import (
    FaceVerificationComparisonEvaluationResult,
)


class JsonLinesFaceVerificationEvaluationResultWriter(
    BaseFaceVerificationEvaluationResultWriter
):
    SCHEMA_VERSION = 1

    def __init__(
        self,
        output_path: Path,
    ) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.output_file: TextIO = self.output_path.open(
            mode="w",
            encoding="utf-8",
        )

    def write_evaluation_result(
        self,
        evaluation_result: FaceVerificationComparisonEvaluationResult,
    ) -> None:
        comparison_sample = evaluation_result.comparison_sample

        evaluation_record = {
            "schema_version": self.SCHEMA_VERSION,
            "dataset_name": evaluation_result.dataset_name,
            "comparison_id": comparison_sample.comparison_id,
            "reference_subject_id": (comparison_sample.reference_subject_id),
            "candidate_subject_id": (comparison_sample.candidate_subject_id),
            "reference_image_path": str(comparison_sample.reference_image_path),
            "candidate_image_path": str(comparison_sample.candidate_image_path),
            "is_genuine": comparison_sample.is_genuine,
            "model_name": evaluation_result.model_name,
            "status": evaluation_result.status.value,
            "processing_time_milliseconds": (
                evaluation_result.processing_time_milliseconds
            ),
            "similarity_score": evaluation_result.similarity_score,
            "failure_detail": evaluation_result.failure_detail,
        }

        serialized_record = json.dumps(
            evaluation_record,
            ensure_ascii=False,
        )

        self.output_file.write(serialized_record)
        self.output_file.write("\n")
        self.output_file.flush()

    def close(self) -> None:
        if not self.output_file.closed:
            self.output_file.close()
