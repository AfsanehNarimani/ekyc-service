import pytest

from core.liveness_action import LivenessAction
from liveness.expected_action_policy import ExpectedActionPolicy


def make_policy() -> ExpectedActionPolicy:
    return ExpectedActionPolicy(
        required_actions=[
            LivenessAction.CENTER,
            LivenessAction.LEFT,
            LivenessAction.RIGHT,
            LivenessAction.UP,
            LivenessAction.DOWN,
        ]
    )


def test_expected_action_policy_accepts_required_actions():
    policy = make_policy()

    policy.validate(
        [
            LivenessAction.CENTER,
            LivenessAction.LEFT,
            LivenessAction.RIGHT,
            LivenessAction.UP,
            LivenessAction.DOWN,
        ]
    )


def test_expected_action_policy_rejects_missing_action():
    policy = make_policy()

    with pytest.raises(ValueError):
        policy.validate(
            [
                LivenessAction.CENTER,
                LivenessAction.LEFT,
                LivenessAction.RIGHT,
                LivenessAction.DOWN,
            ]
        )


def test_expected_action_policy_rejects_wrong_order():
    policy = make_policy()

    with pytest.raises(ValueError):
        policy.validate(
            [
                LivenessAction.CENTER,
                LivenessAction.RIGHT,
                LivenessAction.LEFT,
                LivenessAction.UP,
                LivenessAction.DOWN,
            ]
        )
