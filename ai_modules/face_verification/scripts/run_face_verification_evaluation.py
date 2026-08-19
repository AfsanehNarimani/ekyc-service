import argparse
import csv
from pathlib import Path

from ai_modules.face_verification.evaluations.csv_face_verification_evaluation_dataset import (
    CsvFaceVerificationEvaluationDataset,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_evaluator import (
    FaceVerificationComparisonEvaluator,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_status import (
    FaceVerificationComparisonStatus,
)
from ai_modules.face_verification.evaluations.face_verification_dataset_evaluator import (
    FaceVerificationDatasetEvaluator,
)
from ai_modules.face_verification.evaluations.face_verification_metric_curve_calculator import (
    FaceVerificationMetricCurveCalculator,
)
from ai_modules.face_verification.evaluations.face_verification_metric_pair_calculator import (
    FaceVerificationMetricPairCalculator,
)
from ai_modules.face_verification.evaluations.face_verification_metric_operating_point import (
    FaceVerificationMetricOperatingPoint,
)
from ai_modules.face_verification.evaluations.face_verification_threshold_candidate_generator import (
    FaceVerificationThresholdCandidateGenerator,
)
from ai_modules.face_verification.evaluations.json_lines_face_verification_evaluation_result_writer import (
    JsonLinesFaceVerificationEvaluationResultWriter,
)
from api.dependencies import (
    get_face_detection_service,
    get_face_embedder,
    get_face_embedding_similarity_calculator,
    get_face_preprocessor,
    get_image_file_loader,
)
from config import settings


LEVEL_TO_TARGET_FMR = {
    1: 0.01,
    2: 0.001,
    3: 0.0001,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate face verification comparisons and calculate "
            "thresholds for verification levels."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
    )
    parser.add_argument(
        "--manifest-path",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def select_operating_point(
    operating_points: tuple[FaceVerificationMetricOperatingPoint, ...],
    target_false_match_rate: float,
) -> FaceVerificationMetricOperatingPoint:
    eligible_operating_points = [
        operating_point
        for operating_point in operating_points
        if (operating_point.metric_pair.false_match_rate <= target_false_match_rate)
    ]

    return min(
        eligible_operating_points,
        key=lambda operating_point: operating_point.threshold,
    )


def write_metric_curve(
    output_path: Path,
    operating_points: tuple[FaceVerificationMetricOperatingPoint, ...],
) -> None:
    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "threshold",
                "false_match_rate",
                "true_match_rate",
                "false_non_match_rate",
            ],
        )
        writer.writeheader()

        for operating_point in operating_points:
            metric_pair = operating_point.metric_pair

            writer.writerow(
                {
                    "threshold": operating_point.threshold,
                    "false_match_rate": metric_pair.false_match_rate,
                    "true_match_rate": metric_pair.true_match_rate,
                    "false_non_match_rate": (1.0 - metric_pair.true_match_rate),
                }
            )


def write_verification_levels(
    output_path: Path,
    operating_points: tuple[FaceVerificationMetricOperatingPoint, ...],
    genuine_comparison_count: int,
    impostor_comparison_count: int,
) -> None:
    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "level",
                "target_false_match_rate",
                "threshold",
                "observed_false_match_rate",
                "true_match_rate",
                "false_non_match_rate",
                "genuine_comparison_count",
                "impostor_comparison_count",
            ],
        )
        writer.writeheader()

        for level, target_false_match_rate in LEVEL_TO_TARGET_FMR.items():
            selected_operating_point = select_operating_point(
                operating_points=operating_points,
                target_false_match_rate=target_false_match_rate,
            )
            metric_pair = selected_operating_point.metric_pair

            writer.writerow(
                {
                    "level": level,
                    "target_false_match_rate": (target_false_match_rate),
                    "threshold": selected_operating_point.threshold,
                    "observed_false_match_rate": (metric_pair.false_match_rate),
                    "true_match_rate": metric_pair.true_match_rate,
                    "false_non_match_rate": (1.0 - metric_pair.true_match_rate),
                    "genuine_comparison_count": (genuine_comparison_count),
                    "impostor_comparison_count": (impostor_comparison_count),
                }
            )


def main() -> None:
    arguments = parse_arguments()

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_dataset = CsvFaceVerificationEvaluationDataset(
        dataset_name=arguments.dataset_name,
        manifest_path=arguments.manifest_path,
    )

    comparison_evaluator = FaceVerificationComparisonEvaluator(
        model_name=settings.default_face_model,
        image_file_loader=get_image_file_loader(),
        face_detection_service=get_face_detection_service(),
        face_preprocessor=get_face_preprocessor(),
        face_embedder=get_face_embedder(),
        similarity_calculator=(get_face_embedding_similarity_calculator()),
    )

    dataset_evaluator = FaceVerificationDatasetEvaluator(
        comparison_evaluator=comparison_evaluator,
    )

    genuine_similarity_scores: list[float] = []
    impostor_similarity_scores: list[float] = []

    total_comparison_count = 0
    successful_comparison_count = 0

    comparison_results_path = arguments.output_directory / "comparison_results.jsonl"

    with JsonLinesFaceVerificationEvaluationResultWriter(
        output_path=comparison_results_path,
    ) as result_writer:
        for evaluation_result in dataset_evaluator.iter_comparison_evaluation_results(
            evaluation_dataset=evaluation_dataset,
        ):
            total_comparison_count += 1
            result_writer.write_evaluation_result(
                evaluation_result,
            )

            if evaluation_result.status is not FaceVerificationComparisonStatus.SUCCESS:
                continue

            successful_comparison_count += 1

            similarity_score = evaluation_result.similarity_score

            if similarity_score is None:
                raise RuntimeError("Successful comparison has no similarity score.")

            if evaluation_result.comparison_sample.is_genuine:
                genuine_similarity_scores.append(
                    similarity_score,
                )
            else:
                impostor_similarity_scores.append(
                    similarity_score,
                )

    if not genuine_similarity_scores:
        raise ValueError("No successful genuine comparisons were available.")

    if not impostor_similarity_scores:
        raise ValueError("No successful impostor comparisons were available.")

    threshold_generator = FaceVerificationThresholdCandidateGenerator()

    thresholds = threshold_generator.generate(
        genuine_similarity_scores=genuine_similarity_scores,
        impostor_similarity_scores=impostor_similarity_scores,
    )

    metric_curve_calculator = FaceVerificationMetricCurveCalculator(
        metric_pair_calculator=FaceVerificationMetricPairCalculator(),
    )

    operating_points = metric_curve_calculator.calculate_for_thresholds(
        genuine_similarity_scores=genuine_similarity_scores,
        impostor_similarity_scores=impostor_similarity_scores,
        thresholds=thresholds,
    )

    write_metric_curve(
        output_path=(arguments.output_directory / "metric_curve.csv"),
        operating_points=operating_points,
    )

    write_verification_levels(
        output_path=(arguments.output_directory / "verification_levels.csv"),
        operating_points=operating_points,
        genuine_comparison_count=len(genuine_similarity_scores),
        impostor_comparison_count=len(impostor_similarity_scores),
    )

    print(
        {
            "model_name": settings.default_face_model,
            "total_comparisons": total_comparison_count,
            "successful_comparisons": successful_comparison_count,
            "failed_comparisons": (
                total_comparison_count - successful_comparison_count
            ),
            "genuine_comparisons": len(genuine_similarity_scores),
            "impostor_comparisons": len(impostor_similarity_scores),
            "output_directory": str(arguments.output_directory),
        }
    )


if __name__ == "__main__":
    main()
