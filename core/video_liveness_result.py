from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoLivenessResult:
    passed: bool
    strictness_level: int
    processed_frames: int
    skipped_frames: int
    min_required_frames: int
    pose_variation_score: float
    min_pose_variation_score: float
    reason: str
