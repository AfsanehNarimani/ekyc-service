from abc import ABC, abstractmethod

from core.passive_liveness_result import PassiveLivenessResult


class BasePassiveLivenessChecker(ABC):
    @abstractmethod
    def check(
        self,
        video_analysis,
        strictness_level: int,
    ) -> PassiveLivenessResult:
        pass
