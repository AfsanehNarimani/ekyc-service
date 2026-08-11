import pytest

from core.exceptions import ServiceCapacityExceededError
from services.verification_capacity_limiter import (
    VerificationCapacityLimiter,
)


def test_capacity_limiter_rejects_request_when_full():
    limiter = VerificationCapacityLimiter(max_concurrent_verifications=1)

    with limiter.acquire(), pytest.raises(ServiceCapacityExceededError):
        with limiter.acquire():
            pass


def test_capacity_limiter_releases_slot_after_success():
    limiter = VerificationCapacityLimiter(max_concurrent_verifications=1)

    with limiter.acquire():
        pass

    with limiter.acquire():
        pass


def test_capacity_limiter_releases_slot_after_error():
    limiter = VerificationCapacityLimiter(max_concurrent_verifications=1)

    with pytest.raises(RuntimeError), limiter.acquire():
        raise RuntimeError("test error")

    with limiter.acquire():
        pass


def test_capacity_limiter_rejects_invalid_limit():
    with pytest.raises(
        ValueError,
        match="must be greater than zero",
    ):
        VerificationCapacityLimiter(max_concurrent_verifications=0)
