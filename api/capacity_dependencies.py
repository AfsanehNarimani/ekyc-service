from collections.abc import Iterator
from functools import lru_cache

from fastapi import Depends

from config import settings
from services.verification_capacity_limiter import (
    VerificationCapacityLimiter,
)


@lru_cache
def get_verification_capacity_limiter() -> VerificationCapacityLimiter:
    return VerificationCapacityLimiter(
        max_concurrent_verifications=(settings.max_concurrent_verifications)
    )


def acquire_verification_capacity(
    limiter: VerificationCapacityLimiter = Depends(get_verification_capacity_limiter),
) -> Iterator[None]:
    with limiter.acquire():
        yield
