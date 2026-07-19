from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnrollmentResult:
    person_id: str
    model_name: str
    reference_image_path: str
    embedding_path: str
