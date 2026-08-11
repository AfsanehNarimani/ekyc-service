from abc import ABC, abstractmethod

from ai_modules.face_detection.domain.analyzed_video_frame import (
    VideoFrameAnalysisResult,
)
from core.passive_liveness_result import PassiveLivenessResult


class BasePassiveLivenessChecker(ABC):
    @abstractmethod
    def check(
        self,
        video_analysis: VideoFrameAnalysisResult,
        strictness_level: int,
    ) -> PassiveLivenessResult:
        pass
