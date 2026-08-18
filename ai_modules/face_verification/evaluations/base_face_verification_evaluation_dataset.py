from abc import ABC, abstractmethod
from collections.abc import Iterator

from ai_modules.face_verification.evaluations.face_verification_comparison_sample import (
    FaceVerificationComparisonSample,
)


class BaseFaceVerificationEvaluationDataset(ABC):
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def iter_comparison_samples(self) -> Iterator[FaceVerificationComparisonSample]:
        raise NotImplementedError
