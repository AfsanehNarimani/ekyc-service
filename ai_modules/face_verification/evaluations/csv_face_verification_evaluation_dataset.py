import csv
from collections.abc import Iterator
from pathlib import Path

from ai_modules.face_verification.evaluations.base_face_verification_evaluation_dataset import (
    BaseFaceVerificationEvaluationDataset,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_sample import (
    FaceVerificationComparisonSample,
)


class CsvFaceVerificationEvaluationDataset(BaseFaceVerificationEvaluationDataset):
    def __init__(
        self,
        dataset_name: str,
        manifest_path: Path,
    ) -> None:
        self._dataset_name = dataset_name
        self.manifest_path = manifest_path

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def iter_comparison_samples(
        self,
    ) -> Iterator[FaceVerificationComparisonSample]:
        with self.manifest_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as manifest_file:
            reader = csv.DictReader(manifest_file)

            for row_number, row in enumerate(
                reader,
                start=2,
            ):
                yield FaceVerificationComparisonSample(
                    comparison_id=row["comparison_id"],
                    reference_subject_id=(row["reference_subject_id"]),
                    candidate_subject_id=(row["candidate_subject_id"]),
                    reference_image_path=self._resolve_image_path(
                        row["reference_image_path"]
                    ),
                    candidate_image_path=self._resolve_image_path(
                        row["candidate_image_path"]
                    ),
                    is_genuine=self._parse_is_genuine(
                        value=row["is_genuine"],
                        row_number=row_number,
                    ),
                )

    def _resolve_image_path(
        self,
        path_value: str,
    ) -> Path:
        image_path = Path(path_value)

        if image_path.is_absolute():
            return image_path

        return self.manifest_path.parent / image_path

    @staticmethod
    def _parse_is_genuine(
        value: str,
        row_number: int,
    ) -> bool:
        normalized_value = value.strip().lower()

        if normalized_value == "true":
            return True

        if normalized_value == "false":
            return False

        raise ValueError(
            f"Invalid is_genuine value at CSV row "
            f"{row_number}: {value!r}. "
            "Expected 'true' or 'false'."
        )
