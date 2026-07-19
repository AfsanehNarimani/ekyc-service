from liveness.passive_liveness_threshold_policy import (
    PassiveLivenessThresholdPolicy,
)
from services.stub_passive_liveness_checker import StubPassiveLivenessChecker


def test_stub_passive_liveness_checker_returns_passed_result(tmp_path):
    checker = StubPassiveLivenessChecker(
        threshold_policy=PassiveLivenessThresholdPolicy(
            level_to_threshold={
                2: 0.6,
            }
        )
    )

    result = checker.check(
        video_path=tmp_path / "video.mp4",
        strictness_level=2,
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.threshold == 0.6
    assert result.model_name == "stub_passive_liveness"
    assert result.reason == "stub_passive_liveness_passed"
