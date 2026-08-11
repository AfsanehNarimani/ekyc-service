import pytest

from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_action_type import HeadPoseActionType
from liveness.expected_action_policy import ExpectedActionPolicy


def make_policy() -> ExpectedActionPolicy:
    return ExpectedActionPolicy(
        required_actions=[
            HeadPoseActionType.CENTER,
            HeadPoseActionType.LEFT,
            HeadPoseActionType.RIGHT,
            HeadPoseActionType.UP,
            HeadPoseActionType.DOWN,
        ]
    )


def test_expected_action_policy_accepts_required_actions():
    policy = make_policy()

    policy.validate(
        [
            HeadPoseActionType.CENTER,
            HeadPoseActionType.LEFT,
            HeadPoseActionType.RIGHT,
            HeadPoseActionType.UP,
            HeadPoseActionType.DOWN,
        ]
    )


def test_expected_action_policy_rejects_missing_action():
    policy = make_policy()

    with pytest.raises(ValueError):
        policy.validate(
            [
                HeadPoseActionType.CENTER,
                HeadPoseActionType.LEFT,
                HeadPoseActionType.RIGHT,
                HeadPoseActionType.DOWN,
            ]
        )


def test_expected_action_policy_rejects_wrong_order():
    policy = make_policy()

    with pytest.raises(ValueError):
        policy.validate(
            [
                HeadPoseActionType.CENTER,
                HeadPoseActionType.RIGHT,
                HeadPoseActionType.LEFT,
                HeadPoseActionType.UP,
                HeadPoseActionType.DOWN,
            ]
        )
