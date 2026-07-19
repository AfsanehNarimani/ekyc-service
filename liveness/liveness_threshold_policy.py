from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LivenessThresholds:
    min_required_frames: int
    min_pose_variation_score: float


class LivenessThresholdPolicy:
    def __init__(
        self,
        level_to_thresholds: dict[int, LivenessThresholds],
    ) -> None:
        self.level_to_thresholds = level_to_thresholds

    def get_thresholds(
        self,
        strictness_level: int,
    ) -> LivenessThresholds:
        thresholds = self.level_to_thresholds.get(strictness_level)

        if thresholds is None:
            raise ValueError(
                f"Unsupported liveness strictness level: {strictness_level}"
            )

        return thresholds
