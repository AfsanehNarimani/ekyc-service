from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImageVerificationResult:
    person_id: str
    is_match: bool
    score: float
    threshold: float
    model_name: str
