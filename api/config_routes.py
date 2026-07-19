from fastapi import APIRouter, Depends

from api.config_schemas import VerificationConfigResponse
from api.dependencies import get_verification_config_service
from services.verification_config_service import VerificationConfigService

router = APIRouter(
    prefix="/api",
    tags=["Verification Config"],
)


@router.get(
    "/verification-config",
    response_model=VerificationConfigResponse,
)
def get_verification_config(
    service: VerificationConfigService = Depends(get_verification_config_service),
):
    return service.get_config()
