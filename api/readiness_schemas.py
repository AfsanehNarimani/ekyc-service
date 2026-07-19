from pydantic import BaseModel


class ReadinessCheckResponse(BaseModel):
    name: str
    passed: bool
    message: str


class ReadinessResponse(BaseModel):
    ok: bool
    ready: bool
    checks: list[ReadinessCheckResponse]
