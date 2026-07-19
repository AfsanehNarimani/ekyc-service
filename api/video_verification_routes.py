from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import (
    get_video_verification_orchestrator_service,
    get_video_verification_response_mapper,
)
from api.video_verification_response_mapper import VideoVerificationResponseMapper
from api.video_verification_schemas import (
    VerifyVideoRequest,
    VerifyVideoResponse,
)
from core.exceptions import (
    InvalidVideoError,
    MediaLoadError,
    NoUsableVideoFrameError,
)
from services.video_verification_orchestrator_service import (
    VideoVerificationOrchestratorService,
)

router = APIRouter(
    prefix="/api",
    tags=["Video Verification"],
)


@router.post(
    "/verify-video",
    response_model=VerifyVideoResponse,
)
def verify_video(
    request: VerifyVideoRequest,
    service: VideoVerificationOrchestratorService = Depends(
        get_video_verification_orchestrator_service
    ),
    response_mapper: VideoVerificationResponseMapper = Depends(
        get_video_verification_response_mapper
    ),
):
    try:
        result = service.verify(
            challenge_id=request.challenge_id,
            expected_actions=request.expected_actions,
            national_id_image_uri=request.national_id_image_uri,
            video_uri=request.video_uri,
            face_match_level=request.face_match_level,
            liveness_level=request.liveness_level,
        )

        return response_mapper.to_response(result)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"Media file not found: {error}",
        ) from error

    except MediaLoadError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except InvalidVideoError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except NoUsableVideoFrameError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
