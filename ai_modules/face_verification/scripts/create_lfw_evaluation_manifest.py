import argparse
import csv
from pathlib import Path


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
        "--output-path",
        required=True,
        type=Path,
    )
    return parser.parse_args()


def create_image_path(
    images_directory: Path,
    subject_id: str,
    image_number: int,
) -> Path:
    return images_directory / subject_id / f"{subject_id}_{image_number:04d}.jpg"


def main() -> None:
    arguments = parse_arguments()

    output_path = arguments.output_path.resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    images_directory = arguments.images_directory.resolve()

    pair_lines = arguments.pairs_path.read_text(
        encoding="utf-8",
    ).splitlines()[1:]

    genuine_count = 0
    impostor_count = 0

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "comparison_id",
                "reference_subject_id",
                "candidate_subject_id",
                "reference_image_path",
                "candidate_image_path",
                "is_genuine",
            ],
        )
        writer.writeheader()

        for comparison_number, pair_line in enumerate(
            pair_lines,
            start=1,
        ):
            columns = pair_line.split()

            if len(columns) == 3:
                reference_subject_id = columns[0]
                candidate_subject_id = columns[0]
                reference_image_number = int(columns[1])
                candidate_image_number = int(columns[2])
                is_genuine = True
                genuine_count += 1

            elif len(columns) == 4:
                reference_subject_id = columns[0]
                reference_image_number = int(columns[1])
                candidate_subject_id = columns[2]
                candidate_image_number = int(columns[3])
                is_genuine = False
                impostor_count += 1

            else:
                raise ValueError(f"Invalid pair: {pair_line!r}")

            reference_image_path = create_image_path(
                images_directory=images_directory,
                subject_id=reference_subject_id,
                image_number=reference_image_number,
            )
            candidate_image_path = create_image_path(
                images_directory=images_directory,
                subject_id=candidate_subject_id,
                image_number=candidate_image_number,
            )

            if not reference_image_path.is_file():
                raise FileNotFoundError(reference_image_path)

            if not candidate_image_path.is_file():
                raise FileNotFoundError(candidate_image_path)

            writer.writerow(
                {
                    "comparison_id": (f"lfw-{comparison_number:04d}"),
                    "reference_subject_id": (reference_subject_id),
                    "candidate_subject_id": (candidate_subject_id),
                    "reference_image_path": str(
                        reference_image_path.relative_to(output_path.parent)
                    ),
                    "candidate_image_path": str(
                        candidate_image_path.relative_to(output_path.parent)
                    ),
                    "is_genuine": (str(is_genuine).lower()),
                }
            )

    print(
        {
            "genuine": genuine_count,
            "impostor": impostor_count,
            "total": genuine_count + impostor_count,
            "output": str(output_path),
        }
    )


if __name__ == "__main__":
    main()
