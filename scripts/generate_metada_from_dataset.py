import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


def extract_class_from_samples(file_relative_path) -> str:
    parts = file_relative_path.split("/")
    sample_class = parts[1] if len(parts) > 1 else ""
    return sample_class


def generate_metadata(
    data_root_dir: Path,
    relative_dataset_path2data_root_dir: Path,
) -> None:
    dataset_dir = data_root_dir / relative_dataset_path2data_root_dir
    samples_dir = dataset_dir / "samples"
    metadata_file = dataset_dir / "metadata.csv"

    records = []

    for root, dirs, files in os.walk(samples_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file in files:
            if file.startswith("."):
                continue

            full_path = Path(root) / file

            file_relative_path = full_path.relative_to(dataset_dir).as_posix()

            sample_class = extract_class_from_samples(file_relative_path)

            records.append(
                {
                    "file_name": file_relative_path,
                    "class": sample_class,
                    "source": "",
                    "active_v1": True,
                }
            )

    df = pd.DataFrame(records)
    df = df.sort_values("file_name").reset_index(drop=True)

    if metadata_file.exists():
        backup_file = dataset_dir / "metadata_backup.csv"
        shutil.copy2(metadata_file, backup_file)
        print(f"Backed up existing metadata to {backup_file}")

    df.to_csv(metadata_file, index=False)

    print(f"Saved {len(df)} samples to {metadata_file}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root-dir",
        type=Path,
        default=Path.home() / "data",
        help="Root directory containing datasets.",
    )

    parser.add_argument(
        "--relative-dataset-path2data_root_dir",
        type=Path,
        default=Path("ekyc/passive_liveness"),
        help="Dataset path relative to the data root directory.",
    )

    args = parser.parse_args()

    generate_metadata(
        data_root_dir=args.data_root_dir,
        relative_dataset_path2data_root_dir=args.relative_dataset_path2data_root_dir,
    )


if __name__ == "__main__":
    main()
