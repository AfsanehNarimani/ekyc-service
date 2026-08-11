from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_action_type import (
    HeadPoseActionType,
)


class ExpectedActionPolicy:
    def __init__(
        self,
        required_actions: list[HeadPoseActionType],
    ) -> None:
        self.required_actions = required_actions

    def validate(
        self,
        expected_actions: list[HeadPoseActionType],
    ) -> None:
        # TODO accept all permutations after center
        if expected_actions != self.required_actions:
            raise ValueError(
                "Invalid expected_actions. "
                f"Expected exactly: {[action.value for action in self.required_actions]}"
            )
