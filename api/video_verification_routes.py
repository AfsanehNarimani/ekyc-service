from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.dependencies import (
    get_error_response_factory,
    get_video_verification_orchestrator_service,
    get_video_verification_response_mapper,
)
from api.error_response_factory import ErrorResponseFactory
from api.error_schemas import ErrorResponse
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
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
def verify_video(
    request: VerifyVideoRequest,
    service: VideoVerificationOrchestratorService = Depends(
        get_video_verification_orchestrator_service
    ),
    response_mapper: VideoVerificationResponseMapper = Depends(
        get_video_verification_response_mapper
    ),
    error_response_factory: ErrorResponseFactory = Depends(
        get_error_response_factory
    ),
) -> VerifyVideoResponse | JSONResponse:
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
        return error_response_factory.build(
            status_code=400,
            code="invalid_request",
            message=str(error),
        )

    except FileNotFoundError as error:
        return error_response_factory.build(
            status_code=404,
            code="media_file_not_found",
            message=f"Media file not found: {error}",
        )

    except MediaLoadError as error:
        return error_response_factory.build(
            status_code=422,
            code="media_load_failed",
            message=str(error),
        )

    except InvalidVideoError as error:
        return error_response_factory.build(
            status_code=422,
            code="invalid_video",
            message=str(error),
        )

    except NoUsableVideoFrameError as error:
        return error_response_factory.build(
            status_code=422,
            code="no_usable_video_frame",
            message=str(error),
        )
