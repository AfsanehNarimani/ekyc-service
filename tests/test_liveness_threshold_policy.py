import pytest

from liveness.liveness_threshold_policy import (
    LivenessThresholdPolicy,
    LivenessThresholds,
)


def test_liveness_threshold_policy_returns_thresholds():
    policy = LivenessThresholdPolicy(
        level_to_thresholds={
            2: LivenessThresholds(
                min_required_frames=5,
                min_pose_variation_score=18.0,
            )
        }
    )

    thresholds = policy.get_thresholds(2)

    assert thresholds.min_required_frames == 5
    assert thresholds.min_pose_variation_score == 18.0


def test_liveness_threshold_policy_rejects_unknown_level():
    policy = LivenessThresholdPolicy(level_to_thresholds={})

    with pytest.raises(ValueError):
        policy.get_thresholds(9)
