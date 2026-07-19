import pytest

from face.face_match_threshold_policy import FaceMatchThresholdPolicy


def test_face_match_threshold_policy_returns_threshold():
    policy = FaceMatchThresholdPolicy(
        level_to_threshold={
            1: 0.45,
            2: 0.55,
            3: 0.65,
        }
    )

    assert policy.get_threshold(1) == 0.45
    assert policy.get_threshold(2) == 0.55
    assert policy.get_threshold(3) == 0.65


def test_face_match_threshold_policy_rejects_unknown_level():
    policy = FaceMatchThresholdPolicy(
        level_to_threshold={
            1: 0.45,
        }
    )

    with pytest.raises(ValueError):
        policy.get_threshold(4)
