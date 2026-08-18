import json
from pathlib import Path

from ai_modules.face_verification.evaluations.face_verification_comparison_evaluation_result import (
    FaceVerificationComparisonEvaluationResult,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_sample import (
    FaceVerificationComparisonSample,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_status import (
    FaceVerificationComparisonStatus,
)
from ai_modules.face_verification.evaluations.json_lines_face_verification_evaluation_result_writer import (
    JsonLinesFaceVerificationEvaluationResultWriter,
)


def create_successful_evaluation_result(
    comparison_id: str = "comparison-001",
) -> FaceVerificationComparisonEvaluationResult:
    comparison_sample = FaceVerificationComparisonSample(
        comparison_id=comparison_id,
        reference_subject_id="subject-001",
        candidate_subject_id="subject-001",
        reference_image_path=Path("images/reference.jpg"),
        candidate_image_path=Path("images/candidate.jpg"),
        is_genuine=True,
    )

    return FaceVerificationComparisonEvaluationResult(
        dataset_name="internal-ekyc",
        comparison_sample=comparison_sample,
        model_name="edgeface",
        status=FaceVerificationComparisonStatus.SUCCESS,
        processing_time_milliseconds=42.7,
        similarity_score=0.82,
    )


def create_failed_evaluation_result() -> FaceVerificationComparisonEvaluationResult:
    comparison_sample = FaceVerificationComparisonSample(
        comparison_id="comparison-failed-001",
        reference_subject_id="subject-001",
        candidate_subject_id="subject-002",
        reference_image_path=Path("images/reference.jpg"),
        candidate_image_path=Path("images/candidate.jpg"),
        is_genuine=False,
    )

    return FaceVerificationComparisonEvaluationResult(
        dataset_name="internal-ekyc",
        comparison_sample=comparison_sample,
        model_name="edgeface",
        status=(FaceVerificationComparisonStatus.CANDIDATE_FACE_NOT_DETECTED),
        processing_time_milliseconds=18.4,
        similarity_score=None,
        failure_detail="No face was detected in candidate image.",
    )


def test_write_face_verification_evaluation_result_as_json_line(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "results" / "evaluation.jsonl"

    with JsonLinesFaceVerificationEvaluationResultWriter(
        output_path=output_path,
    ) as result_writer:
        result_writer.write_evaluation_result(
            create_successful_evaluation_result(),
        )

    written_lines = output_path.read_text(
        encoding="utf-8",
    ).splitlines()

    assert len(written_lines) == 1

    written_record = json.loads(written_lines[0])

    assert written_record == {
        "schema_version": 1,
        "dataset_name": "internal-ekyc",
        "comparison_id": "comparison-001",
        "reference_subject_id": "subject-001",
        "candidate_subject_id": "subject-001",
        "reference_image_path": "images/reference.jpg",
        "candidate_image_path": "images/candidate.jpg",
        "is_genuine": True,
        "model_name": "edgeface",
        "status": "success",
        "processing_time_milliseconds": 42.7,
        "similarity_score": 0.82,
        "failure_detail": None,
    }


def test_write_multiple_evaluation_results_incrementally(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "evaluation.jsonl"

    with JsonLinesFaceVerificationEvaluationResultWriter(
        output_path=output_path,
    ) as result_writer:
        result_writer.write_evaluation_result(
            create_successful_evaluation_result(
                comparison_id="comparison-001",
            ),
        )

        written_lines_after_first_result = output_path.read_text(
            encoding="utf-8",
        ).splitlines()

        assert len(written_lines_after_first_result) == 1

        result_writer.write_evaluation_result(
            create_successful_evaluation_result(
                comparison_id="comparison-002",
            ),
        )

        written_lines_after_second_result = output_path.read_text(
            encoding="utf-8",
        ).splitlines()

        assert len(written_lines_after_second_result) == 2

    written_records = [
        json.loads(line)
        for line in output_path.read_text(
            encoding="utf-8",
        ).splitlines()
    ]

    assert [record["comparison_id"] for record in written_records] == [
        "comparison-001",
        "comparison-002",
    ]


def test_write_failed_evaluation_result_as_json_line(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "failed-evaluation.jsonl"

    with JsonLinesFaceVerificationEvaluationResultWriter(
        output_path=output_path,
    ) as result_writer:
        result_writer.write_evaluation_result(
            create_failed_evaluation_result(),
        )

    written_record = json.loads(
        output_path.read_text(
            encoding="utf-8",
        ).strip()
    )

    assert written_record["comparison_id"] == "comparison-failed-001"
    assert written_record["status"] == "candidate_face_not_detected"
    assert written_record["similarity_score"] is None
    assert written_record["failure_detail"] == (
        "No face was detected in candidate image."
    )
