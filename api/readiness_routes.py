from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from api.dependencies import get_readiness_service
from api.readiness_schemas import ReadinessResponse
from services.readiness_service import ReadinessService

router = APIRouter(
    tags=["Health"],
)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
)
def readiness(
    service: ReadinessService = Depends(get_readiness_service),
) -> ReadinessResponse | JSONResponse:
    result = service.check()

    response = ReadinessResponse(
        ok=result.ready,
        ready=result.ready,
        checks=[
            {
                "name": check.name,
                "passed": check.passed,
                "message": check.message,
            }
            for check in result.checks
        ],
    )

    if result.ready:
        return response

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=response.model_dump(),
    )
