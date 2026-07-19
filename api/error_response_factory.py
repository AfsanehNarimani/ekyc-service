from typing import Any

from fastapi.responses import JSONResponse

from api.error_schemas import ErrorResponse


class ErrorResponseFactory:
    def build(
        self,
        status_code: int,
        code: str,
        message: str,
        trace_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> JSONResponse:
        response = ErrorResponse(
            trace_id=trace_id,
            error={
                "code": code,
                "message": message,
                "details": details or {},
            },
        )

        return JSONResponse(
            status_code=status_code,
            content=response.model_dump(),
        )
