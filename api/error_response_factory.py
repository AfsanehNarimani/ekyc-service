from typing import Any

from fastapi.responses import JSONResponse

from api.error_schemas import ErrorResponse


class ErrorResponseFactory:
    def build(
        self,
        status_code: int,
        code: str,
        message: str,
        correlation_id: str | None = None,
        processing_time_ms: float | None = None,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> JSONResponse:
        response = ErrorResponse(
            correlation_id=correlation_id,
            processing_time_ms=processing_time_ms,
            error={
                "code": code,
                "message": message,
                "details": details or {},
            },
        )

        return JSONResponse(
            status_code=status_code,
            content=response.model_dump(),
            headers=headers,
        )
