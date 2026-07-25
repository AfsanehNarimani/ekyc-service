from abc import ABC, abstractmethod

from core.passive_liveness_result import PassiveLivenessResult
from core.analyzed_video_frame import (
    VideoFrameAnalysisResult,
)


class BasePassiveLivenessChecker(ABC):
    @abstractmethod
    def check(
        self,
        video_analysis: VideoFrameAnalysisResult,
        strictness_level: int,
    ) -> PassiveLivenessResult:
        pass
