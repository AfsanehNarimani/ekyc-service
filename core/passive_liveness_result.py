from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PassiveLivenessResult:
    passed: bool
    score: float
    threshold: float
    model_name: str
    processed_frames: int
    skipped_frames: int
    reason: str
    frame_scores: list[float] | None = None
