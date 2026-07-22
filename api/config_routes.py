from fastapi import APIRouter, Depends

from api.config_schemas import VerificationConfigResponse
from api.dependencies import get_verification_config_service
from services.verification_config_service import VerificationConfigService
from api.authentication_dependencies import authenticate_service
from authentication.authenticated_service import AuthenticatedService
from api.error_schemas import ErrorResponse

router = APIRouter(
    prefix="/api",
    tags=["Verification Config"],
)


@router.get(
    "/verification-config",
    response_model=VerificationConfigResponse,
    responses={
        401: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def get_verification_config(
    _authenticated_service: AuthenticatedService | None = Depends(authenticate_service),
    service: VerificationConfigService = Depends(get_verification_config_service),
) -> VerificationConfigResponse:
    return service.get_config()
