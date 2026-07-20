import pytest

from liveness.passive_liveness_threshold_policy import (
    PassiveLivenessThresholdPolicy,
)


def test_passive_liveness_threshold_policy_returns_threshold():
    policy = PassiveLivenessThresholdPolicy(
        level_to_threshold={
            2: 0.6,
        }
    )

    assert policy.get_threshold(2) == 0.6


def test_passive_liveness_threshold_policy_rejects_unknown_level():
    policy = PassiveLivenessThresholdPolicy(level_to_threshold={})

    with pytest.raises(ValueError):
        policy.get_threshold(9)
