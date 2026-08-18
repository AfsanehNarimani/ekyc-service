from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FaceVerificationComparisonSample:
    comparison_id: str
    reference_subject_id: str
    candidate_subject_id: str
    reference_image_path: Path
    candidate_image_path: Path
    is_genuine: bool
