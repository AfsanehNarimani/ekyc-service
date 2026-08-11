from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoFaceMatchResult:
    is_match: bool
    score: float
    best_score: float
    threshold: float
    model_name: str
    processed_frames: int
    skipped_frames: int
    frame_scores: list[float]
