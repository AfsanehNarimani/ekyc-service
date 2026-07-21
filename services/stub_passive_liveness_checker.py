from core.passive_liveness_result import PassiveLivenessResult
from liveness.base_passive_liveness_checker import BasePassiveLivenessChecker
from liveness.passive_liveness_threshold_policy import (
    PassiveLivenessThresholdPolicy,
)
from core.analyzed_video_frame import VideoFrameAnalysisResult


class StubPassiveLivenessChecker(BasePassiveLivenessChecker):
    def __init__(
        self,
        threshold_policy: PassiveLivenessThresholdPolicy,
    ) -> None:
        self.threshold_policy = threshold_policy

    def check(
        self,
        video_analysis: VideoFrameAnalysisResult,
        strictness_level: int,
    ) -> PassiveLivenessResult:
        threshold = self.threshold_policy.get_threshold(strictness_level)

        return PassiveLivenessResult(
            passed=True,
            score=1.0,
            threshold=threshold,
            model_name="stub_passive_liveness",
            processed_frames=0,
            skipped_frames=0,
            reason="stub_passive_liveness_passed",
            frame_scores=[],
        )
