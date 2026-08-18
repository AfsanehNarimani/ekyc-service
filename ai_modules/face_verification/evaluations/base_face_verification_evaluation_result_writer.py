from abc import ABC, abstractmethod
from types import TracebackType

from ai_modules.face_verification.evaluations.face_verification_comparison_evaluation_result import (
    FaceVerificationComparisonEvaluationResult,
)


class BaseFaceVerificationEvaluationResultWriter(ABC):
    def __enter__(
        self,
    ) -> "BaseFaceVerificationEvaluationResultWriter":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @abstractmethod
    def write_evaluation_result(
        self,
        evaluation_result: FaceVerificationComparisonEvaluationResult,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
