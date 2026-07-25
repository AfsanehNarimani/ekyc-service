from contextlib import contextmanager
from threading import BoundedSemaphore
from typing import Iterator

from core.exceptions import ServiceCapacityExceededError


class VerificationCapacityLimiter:
    def __init__(
        self,
        max_concurrent_verifications: int,
    ) -> None:
        if max_concurrent_verifications <= 0:
            raise ValueError(
                "max_concurrent_verifications must be " "greater than zero."
            )

        self.max_concurrent_verifications = max_concurrent_verifications
        self._semaphore = BoundedSemaphore(value=max_concurrent_verifications)

    @contextmanager
    def acquire(self) -> Iterator[None]:
        acquired = self._semaphore.acquire(blocking=False)

        if not acquired:
            raise ServiceCapacityExceededError("Verification capacity is exhausted.")

        try:
            yield
        finally:
            self._semaphore.release()
