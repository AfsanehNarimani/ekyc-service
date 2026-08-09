import pytest

from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_action import HeadPoseAction
from liveness.expected_action_policy import ExpectedActionPolicy


def make_policy() -> ExpectedActionPolicy:
    return ExpectedActionPolicy(
        required_actions=[
            HeadPoseAction.CENTER,
            HeadPoseAction.LEFT,
            HeadPoseAction.RIGHT,
            HeadPoseAction.UP,
            HeadPoseAction.DOWN,
        ]
    )


def test_expected_action_policy_accepts_required_actions():
    policy = make_policy()

    policy.validate(
        [
            HeadPoseAction.CENTER,
            HeadPoseAction.LEFT,
            HeadPoseAction.RIGHT,
            HeadPoseAction.UP,
            HeadPoseAction.DOWN,
        ]
    )


def test_expected_action_policy_rejects_missing_action():
    policy = make_policy()

    with pytest.raises(ValueError):
        policy.validate(
            [
                HeadPoseAction.CENTER,
                HeadPoseAction.LEFT,
                HeadPoseAction.RIGHT,
                HeadPoseAction.DOWN,
            ]
        )


def test_expected_action_policy_rejects_wrong_order():
    policy = make_policy()

    with pytest.raises(ValueError):
        policy.validate(
            [
                HeadPoseAction.CENTER,
                HeadPoseAction.RIGHT,
                HeadPoseAction.LEFT,
                HeadPoseAction.UP,
                HeadPoseAction.DOWN,
            ]
        )
