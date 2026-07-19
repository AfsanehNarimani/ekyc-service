from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.error_response_factory import ErrorResponseFactory


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    trace_id = await _extract_trace_id(request)

    return ErrorResponseFactory().build(
        status_code=422,
        code="request_validation_failed",
        message="Request validation failed.",
        trace_id=trace_id,
        processing_time_ms=0.0,
        details={
            "errors": _safe_validation_errors(exc),
        },
    )


async def _extract_trace_id(
    request: Request,
) -> str | None:
    try:
        body = await request.json()
    except Exception:
        return None

    if not isinstance(body, dict):
        return None

    trace_id = body.get("trace_id")

    if isinstance(trace_id, str) and trace_id.strip():
        return trace_id

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
