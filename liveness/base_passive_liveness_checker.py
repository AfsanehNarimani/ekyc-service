from abc import ABC, abstractmethod
from pathlib import Path

from core.passive_liveness_result import PassiveLivenessResult


class BasePassiveLivenessChecker(ABC):
    @abstractmethod
    def check(
        self,
        video_path: Path,
        strictness_level: int,
    ) -> PassiveLivenessResult:
        pass
