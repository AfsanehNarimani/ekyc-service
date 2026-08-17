from abc import ABC, abstractmethod
from typing import Sequence


class BaseFaceVerificationThresholdCandidateGenerator(ABC):
    @abstractmethod
    def generate(
            self,
            genuine_similarity_scores: Sequence[float],
            impostor_similarity_scores: Sequence[float],
    ) -> tuple[float, ...]:
        raise NotImplementedError