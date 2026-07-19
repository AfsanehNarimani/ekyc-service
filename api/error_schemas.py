from typing import Any

from pydantic import BaseModel, Field


class ErrorDetailResponse(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    ok: bool = False
    trace_id: str | None = None
    processing_time_ms: float | None = None
    error: ErrorDetailResponse
