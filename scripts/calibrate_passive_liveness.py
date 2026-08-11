from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from api.dependencies import (
    get_passive_liveness_checker,
    get_video_frame_extractor,
    get_face_detection_service,
)
from config import settings
from core.passive_liveness_calibration import (
    PassiveLivenessCalibrationEvaluator,
    PassiveLivenessCalibrationScore,
    PassiveLivenessLabel,
)


@dataclass(frozen=True, slots=True)
class CalibrationManifestSample:
    sample_id: str
    label: PassiveLivenessLabel
    video_path: Path
    device: str
    attack_type: str
    notes: str


@dataclass(frozen=True, slots=True)
class CalibrationSampleResult:
    sample_id: str
    label: str
    score: float
    passed: bool
    configured_threshold: float
    model_name: str
    processed_frames: int
    skipped_frames: int
    reason: str
    device: str
    attack_type: str
    notes: str


def read_manifest(
    manifest_path: Path,
) -> list[CalibrationManifestSample]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    required_columns = {
        "sample_id",
        "label",
        "video_uri",
        "device",
        "attack_type",
        "notes",
    }

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        reader = csv.DictReader(manifest_file)

        available_columns = set(reader.fieldnames or [])

        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                "Missing manifest columns: " + ", ".join(sorted(missing_columns))
            )

        samples = [
            CalibrationManifestSample(
                sample_id=row["sample_id"].strip(),
                label=PassiveLivenessLabel(row["label"].strip()),
                video_path=Path(row["video_uri"].strip()).expanduser(),
                device=row["device"].strip(),
                attack_type=row["attack_type"].strip(),
                notes=row["notes"].strip(),
            )
            for row in reader
        ]

    if not samples:
        raise ValueError("Calibration manifest must not be empty.")

    for sample in samples:
        if not sample.sample_id:
            raise ValueError("sample_id must not be empty.")

        if not sample.video_path.is_file():
            raise FileNotFoundError(sample.video_path)

    return samples


def build_thresholds(
    step: float,
) -> list[float]:
    if not 0.0 < step <= 1.0:
        raise ValueError("threshold step must be between 0.0 and 1.0.")

    thresholds = {
        settings.passive_liveness_threshold_level_1,
        settings.passive_liveness_threshold_level_2,
        settings.passive_liveness_threshold_level_3,
    }

    current = step

    while current < 1.0:
        thresholds.add(round(current, 6))
        current += step

    return sorted(thresholds)


def write_sample_results(
    output_path: Path,
    results: list[CalibrationSampleResult],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(asdict(results[0]).keys()),
        )

        writer.writeheader()

        for result in results:
            writer.writerow(asdict(result))


def write_threshold_metrics(
    output_path: Path,
    metrics,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_rows = [asdict(metric) for metric in metrics]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(metric_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(metric_rows)


def run_calibration(
    manifest_path: Path,
    output_dir: Path,
    strictness_level: int,
    threshold_step: float,
) -> None:
    if settings.passive_liveness_backend != "onnx":
        raise ValueError("Passive liveness calibration requires " "the ONNX backend.")

    samples = read_manifest(
        manifest_path=manifest_path,
    )

    frame_extractor = get_video_frame_extractor()
    face_detection_service = get_face_detection_service()
    checker = get_passive_liveness_checker()

    sample_results: list[CalibrationSampleResult] = []
    calibration_scores: list[PassiveLivenessCalibrationScore] = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        print(f"[{index}/{len(samples)}] " f"processing {sample.sample_id}")

        frames = frame_extractor.extract(
            video_path=sample.video_path,
        )

        video_analysis = face_detection_service.detect_from_frames(
            frames=frames,
        )

        result = checker.check(
            video_analysis=video_analysis,
            strictness_level=strictness_level,
        )

        score = round(
            result.score,
            8,
        )

        sample_results.append(
            CalibrationSampleResult(
                sample_id=sample.sample_id,
                label=sample.label.value,
                score=score,
                passed=result.passed,
                configured_threshold=(result.threshold),
                model_name=result.model_name,
                processed_frames=(result.processed_frames),
                skipped_frames=(result.skipped_frames),
                reason=result.reason,
                device=sample.device,
                attack_type=sample.attack_type,
                notes=sample.notes,
            )
        )

        calibration_scores.append(
            PassiveLivenessCalibrationScore(
                sample_id=sample.sample_id,
                label=sample.label,
                score=score,
            )
        )

    thresholds = build_thresholds(
        step=threshold_step,
    )

    metrics = PassiveLivenessCalibrationEvaluator().evaluate(
        samples=calibration_scores,
        thresholds=thresholds,
    )

    sample_output_path = output_dir / "sample_scores.csv"
    metrics_output_path = output_dir / "threshold_metrics.csv"

    write_sample_results(
        output_path=sample_output_path,
        results=sample_results,
    )
    write_threshold_metrics(
        output_path=metrics_output_path,
        metrics=metrics,
    )

    print(f"sample results: {sample_output_path}")
    print(f"threshold metrics: {metrics_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("calibration/results"),
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=[
            1,
            2,
            3,
        ],
        default=2,
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    run_calibration(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        strictness_level=args.level,
        threshold_step=args.threshold_step,
    )


if __name__ == "__main__":
    main()
