import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from api.dependencies import (
    get_face_detection_service,
    get_face_embedder,
    get_face_embedding_similarity_calculator,
    get_face_preprocessor,
    get_image_file_loader,
)
from config import settings
from ai_modules.face_verification.evaluations.face_verification_metrics import (
    LEVEL_TARGETS,
    calculate_rate,
    select_threshold,
)
from ai_modules.face_verification.domain.face_embedding import FaceEmbedding


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pairs-path",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--images-directory",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--impostor-count",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--embeddings-path",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def load_images_by_subject(
    images_directory: Path,
) -> dict[str, tuple[Path, ...]]:
    return {
        directory.name: tuple(sorted(directory.glob("*.jpg")))
        for directory in sorted(images_directory.iterdir())
        if directory.is_dir()
    }


def image_path(
    images_directory: Path,
    subject_id: str,
    image_number: int,
) -> Path:
    return images_directory / subject_id / f"{subject_id}_{image_number:04d}.jpg"


def load_genuine_pairs(
    pairs_path: Path,
    images_directory: Path,
) -> tuple[tuple[Path, Path], ...]:
    pairs: list[tuple[Path, Path]] = []

    lines = pairs_path.read_text(encoding="utf-8").splitlines()[1:]

    for line in lines:
        columns = line.split()

        if len(columns) != 3:
            continue

        subject_id = columns[0]

        pairs.append(
            (
                image_path(
                    images_directory,
                    subject_id,
                    int(columns[1]),
                ),
                image_path(
                    images_directory,
                    subject_id,
                    int(columns[2]),
                ),
            )
        )

    return tuple(pairs)


def face_area(detection) -> float:
    left, top, right, bottom = detection.bbox
    return (right - left) * (bottom - top)


def extract_embeddings(
    images_by_subject: dict[
        str,
        tuple[Path, ...],
    ],
):
    loader = get_image_file_loader()
    detector = get_face_detection_service().detector
    preprocessor = get_face_preprocessor()
    embedder = get_face_embedder()

    all_image_paths = [path for paths in images_by_subject.values() for path in paths]

    embeddings = {}
    failures = Counter()

    for index, path in enumerate(
        all_image_paths,
        start=1,
    ):
        try:
            image = loader.load(path)
            detections = detector.detect(image)

            if not detections:
                failures["face_not_detected"] += 1
                continue

            primary_face = max(
                detections,
                key=face_area,
            )

            aligned_face = preprocessor.preprocess_with_detection(
                image=image,
                detection=primary_face,
            )

            embeddings[path] = embedder.extract(aligned_face)

        except Exception as error:
            failures[type(error).__name__] += 1

        if index % 500 == 0:
            print(f"Embedded {index}/{len(all_image_paths)} images")

    return embeddings, failures


def save_embeddings(
    output_path: Path,
    embeddings: dict[Path, FaceEmbedding],
    failures: Counter,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = tuple(embeddings)
    face_embeddings = tuple(embeddings.values())

    np.savez_compressed(
        output_path,
        image_paths=np.asarray(
            [str(path) for path in image_paths],
        ),
        vectors=np.stack(
            [embedding.vector for embedding in face_embeddings],
        ),
        model_names=np.asarray(
            [embedding.model_name for embedding in face_embeddings],
        ),
        failure_names=np.asarray(
            list(failures),
        ),
        failure_counts=np.asarray(
            list(failures.values()),
            dtype=np.int64,
        ),
    )


def load_embeddings(
    input_path: Path,
) -> tuple[dict[Path, FaceEmbedding], Counter]:
    with np.load(
        input_path,
        allow_pickle=False,
    ) as data:
        embeddings = {
            Path(image_path): FaceEmbedding(
                vector=vector,
                model_name=str(model_name),
            )
            for image_path, vector, model_name in zip(
                data["image_paths"],
                data["vectors"],
                data["model_names"],
                strict=True,
            )
        }

        failures = Counter(
            {
                str(name): int(count)
                for name, count in zip(
                    data["failure_names"],
                    data["failure_counts"],
                    strict=True,
                )
            }
        )

    return embeddings, failures


def genuine_scores(
    pairs: tuple[tuple[Path, Path], ...],
    embeddings,
) -> tuple[np.ndarray, int]:
    calculator = get_face_embedding_similarity_calculator()

    scores = []
    skipped = 0

    for reference_path, candidate_path in pairs:
        reference = embeddings.get(reference_path)
        candidate = embeddings.get(candidate_path)

        if reference is None or candidate is None:
            skipped += 1
            continue

        scores.append(
            calculator.calculate_similarity(
                reference_embedding=reference,
                candidate_embedding=candidate,
            )
        )

    return (
        np.asarray(scores, dtype=np.float64),
        skipped,
    )


def impostor_scores(
    images_by_subject: dict[
        str,
        tuple[Path, ...],
    ],
    embeddings,
    pair_count: int,
    random_seed: int,
) -> tuple[np.ndarray, int]:
    calculator = get_face_embedding_similarity_calculator()
    random_generator = random.Random(random_seed)

    subject_ids = tuple(images_by_subject)

    all_paths = [path for paths in images_by_subject.values() for path in paths]
    image_id = {path: index for index, path in enumerate(all_paths)}
    image_count = len(all_paths)

    selected_pairs: set[int] = set()
    scores = []
    skipped = 0
    generated = 0

    while generated < pair_count:
        reference_subject, candidate_subject = random_generator.sample(
            subject_ids,
            k=2,
        )

        reference_path = random_generator.choice(images_by_subject[reference_subject])
        candidate_path = random_generator.choice(images_by_subject[candidate_subject])

        first_id, second_id = sorted(
            (
                image_id[reference_path],
                image_id[candidate_path],
            )
        )
        pair_key = first_id * image_count + second_id

        if pair_key in selected_pairs:
            continue

        selected_pairs.add(pair_key)
        generated += 1

        reference = embeddings.get(reference_path)
        candidate = embeddings.get(candidate_path)

        if reference is None or candidate is None:
            skipped += 1
            continue

        scores.append(
            calculator.calculate_similarity(
                reference_embedding=reference,
                candidate_embedding=candidate,
            )
        )

        if generated % 100_000 == 0:
            print(f"Compared {generated} impostor pairs")

    return (
        np.asarray(scores, dtype=np.float64),
        skipped,
    )


def write_levels(
    output_path: Path,
    genuine: np.ndarray,
    impostor: np.ndarray,
) -> tuple[float, ...]:
    thresholds = []

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "level",
                "target_false_match_rate",
                "threshold",
                "observed_false_match_rate",
                "true_match_rate",
                "false_non_match_rate",
                "genuine_comparison_count",
                "impostor_comparison_count",
            ]
        )

        for level, target in LEVEL_TARGETS.items():
            threshold = select_threshold(
                impostor,
                target,
            )
            fmr = calculate_rate(
                impostor,
                threshold,
            )

            tmr = calculate_rate(
                genuine,
                threshold,
            )

            thresholds.append(threshold)

            writer.writerow(
                [
                    level,
                    target,
                    threshold,
                    fmr,
                    tmr,
                    1.0 - tmr,
                    genuine.size,
                    impostor.size,
                ]
            )

    return tuple(thresholds)


def write_curve(
    output_path: Path,
    genuine: np.ndarray,
    impostor: np.ndarray,
    level_thresholds: tuple[float, ...],
) -> None:
    thresholds = np.unique(
        np.concatenate(
            (
                np.linspace(
                    min(
                        genuine.min(),
                        impostor.min(),
                    ),
                    max(
                        genuine.max(),
                        impostor.max(),
                    ),
                    2000,
                ),
                level_thresholds,
            )
        )
    )

    sorted_genuine = np.sort(genuine)
    sorted_impostor = np.sort(impostor)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "threshold",
                "false_match_rate",
                "true_match_rate",
                "false_non_match_rate",
            ]
        )

        for threshold in thresholds:
            false_matches = impostor.size - np.searchsorted(
                sorted_impostor,
                threshold,
            )
            true_matches = genuine.size - np.searchsorted(
                sorted_genuine,
                threshold,
            )

            fmr = false_matches / impostor.size
            tmr = true_matches / genuine.size

            writer.writerow(
                [
                    threshold,
                    fmr,
                    tmr,
                    1.0 - tmr,
                ]
            )


def main() -> None:
    arguments = parse_arguments()

    pairs_path = arguments.pairs_path.resolve()
    images_directory = arguments.images_directory.resolve()
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    images_by_subject = load_images_by_subject(images_directory)
    official_genuine_pairs = load_genuine_pairs(
        pairs_path,
        images_directory,
    )

    embeddings_path = arguments.embeddings_path.resolve()

    if embeddings_path.exists():
        print(f"Loading cached embeddings from {embeddings_path}")

        embeddings, failures = load_embeddings(
            embeddings_path,
        )
    else:
        print("Embedding cache not found. Extracting embeddings...")

        embeddings, failures = extract_embeddings(
            images_by_subject,
        )

        save_embeddings(
            output_path=embeddings_path,
            embeddings=embeddings,
            failures=failures,
        )

        print(f"Saved embeddings to {embeddings_path}")

    genuine, skipped_genuine = genuine_scores(
        official_genuine_pairs,
        embeddings,
    )

    impostor, skipped_impostor = impostor_scores(
        images_by_subject=(images_by_subject),
        embeddings=embeddings,
        pair_count=(arguments.impostor_count),
        random_seed=(arguments.random_seed),
    )

    np.savez_compressed(
        output_directory / "similarity_scores.npz",
        genuine_scores=genuine,
        impostor_scores=impostor,
    )

    level_thresholds = write_levels(
        output_directory / "verification_levels.csv",
        genuine,
        impostor,
    )

    write_curve(
        output_directory / "metric_curve.csv",
        genuine,
        impostor,
        level_thresholds,
    )

    summary = {
        "model_name": (settings.default_face_model),
        "subject_count": len(images_by_subject),
        "image_count": sum(len(paths) for paths in (images_by_subject.values())),
        "successful_embedding_count": len(embeddings),
        "embedding_failures": dict(failures),
        "genuine_score_count": int(genuine.size),
        "impostor_score_count": int(impostor.size),
        "skipped_genuine_pairs": (skipped_genuine),
        "skipped_impostor_pairs": (skipped_impostor),
    }

    (output_directory / "evaluation_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary)


if __name__ == "__main__":
    main()
