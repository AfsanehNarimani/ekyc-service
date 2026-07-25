from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.error_response_factory import ErrorResponseFactory
from authentication.exceptions import (
    AuthenticationConfigurationError,
    ServiceAuthenticationError,
)
from config import settings
from core.exceptions import ServiceCapacityExceededError


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    correlation_id = await extract_correlation_id(request)

    return ErrorResponseFactory().build(
        status_code=422,
        code="request_validation_failed",
        message="Request validation failed.",
        correlation_id=correlation_id,
        processing_time_ms=0.0,
        details={
            "errors": _safe_validation_errors(exc),
        },
    )


async def extract_correlation_id(
    request: Request,
) -> str | None:
    try:
        body = await request.json()
    except Exception:
        return None

    if not isinstance(body, dict):
        return None

    correlation_id = body.get("correlation_id")

    if isinstance(correlation_id, str) and correlation_id.strip():
        return correlation_id

    return None


def _safe_validation_errors(
    exc: RequestValidationError,
) -> list[dict[str, Any]]:
    safe_errors: list[dict[str, Any]] = []

    for error in exc.errors():
        safe_error = {
            "type": error.get("type"),
            "loc": error.get("loc"),
            "msg": error.get("msg"),
        }

        if "ctx" in error:
            safe_error["ctx"] = error["ctx"]

        safe_errors.append(safe_error)

    return safe_errors


async def service_authentication_exception_handler(
    request: Request,
    exc: ServiceAuthenticationError,
) -> JSONResponse:
    correlation_id = await extract_correlation_id(request)

    return ErrorResponseFactory().build(
        status_code=401,
        code="service_authentication_failed",
        message="Service authentication failed.",
        correlation_id=correlation_id,
        processing_time_ms=0.0,
    )


async def authentication_configuration_exception_handler(
    request: Request,
    exc: AuthenticationConfigurationError,
) -> JSONResponse:
    correlation_id = await extract_correlation_id(request)

    return ErrorResponseFactory().build(
        status_code=503,
        code="service_authentication_unavailable",
        message="Service authentication is unavailable.",
        correlation_id=correlation_id,
        processing_time_ms=0.0,
    )


async def service_capacity_exception_handler(
    request: Request,
    exc: ServiceCapacityExceededError,
) -> JSONResponse:
    correlation_id = await extract_correlation_id(request)

    return ErrorResponseFactory().build(
        status_code=503,
        code="verification_capacity_exceeded",
        message=("Verification capacity is temporarily " "unavailable."),
        correlation_id=correlation_id,
        processing_time_ms=0.0,
        headers={
            "Retry-After": str(settings.capacity_retry_after_seconds),
        },
    )
