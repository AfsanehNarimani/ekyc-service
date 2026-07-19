from typing import Any

from pydantic import BaseModel, Field


class ErrorDetailResponse(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    ok: bool = False
    error: ErrorDetailResponse
