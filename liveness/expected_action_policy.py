from core.liveness_action import LivenessAction


class ExpectedActionPolicy:
    def __init__(
        self,
        required_actions: list[LivenessAction],
    ) -> None:
        self.required_actions = required_actions

    def validate(
        self,
        expected_actions: list[LivenessAction],
    ) -> None:
        if expected_actions != self.required_actions:
            raise ValueError(
                "Invalid expected_actions. "
                f"Expected exactly: {[action.value for action in self.required_actions]}"
            )
