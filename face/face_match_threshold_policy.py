class FaceMatchThresholdPolicy:
    def __init__(
        self,
        level_to_threshold: dict[int, float],
    ) -> None:
        self.level_to_threshold = level_to_threshold

    def get_threshold(
        self,
        strictness_level: int,
    ) -> float:
        threshold = self.level_to_threshold.get(strictness_level)

        if threshold is None:
            raise ValueError(
                f"Unsupported face match strictness level: {strictness_level}"
            )

        return threshold
