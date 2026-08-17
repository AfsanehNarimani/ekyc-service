from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.authentication_dependencies import authenticate_service
from api.capacity_dependencies import (
    acquire_verification_capacity,
)
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
from authentication.authenticated_service import AuthenticatedService
from core.exceptions import (
    InvalidVideoError,
    MediaLoadError,
    NoUsableVideoFrameError,
)
from services.head_pose_challenge_orchestrator_service import (
    HeadPoseChallengeOrchestratorService,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Video Verification"],
)


@router.post(
    "/verify-video",
    response_model=VerifyVideoResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {
            "model": ErrorResponse,
            "headers": {
                "Retry-After": {
                    "description": (
                        "Seconds before retrying when "
                        "verification capacity is exhausted."
                    ),
                    "schema": {
                        "type": "integer",
                    },
                },
            },
        },
    },
)
def verify_video(
    request: VerifyVideoRequest,
    _authenticated_service: AuthenticatedService | None = Depends(authenticate_service),
    service: HeadPoseChallengeOrchestratorService = Depends(
        get_video_verification_orchestrator_service
    ),
    _capacity_slot: None = Depends(acquire_verification_capacity),
    response_mapper: VideoVerificationResponseMapper = Depends(
        get_video_verification_response_mapper
    ),
    error_response_factory: ErrorResponseFactory = Depends(get_error_response_factory),
) -> VerifyVideoResponse | JSONResponse:
    correlation_id = request.correlation_id or uuid.uuid4().hex
    challenge_id = request.challenge_id or uuid.uuid4().hex
    started_at = time.perf_counter()

    try:
        result = service.verify(
            correlation_id=correlation_id,
            challenge_id=challenge_id,
            expected_actions=request.expected_actions,
            national_id_image_uri=request.national_id_image_uri,
            video_uri=request.video_uri,
            face_match_level=request.face_match_level,
            liveness_level=request.liveness_level,
        )

        return response_mapper.to_response(result)

    except ValueError as error:
        logger.warning(
            "correlation_id=%s challenge_id=%s error_code=invalid_request message=%s",
            correlation_id,
            request.challenge_id,
            error,
        )
        return error_response_factory.build(
            status_code=400,
            code="invalid_request",
            message=str(error),
            correlation_id=correlation_id,
            processing_time_ms=_elapsed_ms(started_at),
        )

    except FileNotFoundError as error:
        logger.warning(
            "correlation_id=%s challenge_id=%s error_code=media_file_not_found message=%s",
            correlation_id,
            request.challenge_id,
            error,
        )
        return error_response_factory.build(
            status_code=404,
            code="media_file_not_found",
            message=f"Media file not found: {error}",
            correlation_id=correlation_id,
            processing_time_ms=_elapsed_ms(started_at),
        )

    except MediaLoadError as error:
        logger.warning(
            "correlation_id=%s challenge_id=%s error_code=media_load_failed message=%s",
            correlation_id,
            request.challenge_id,
            error,
        )
        return error_response_factory.build(
            status_code=422,
            code="media_load_failed",
            message=str(error),
            correlation_id=correlation_id,
            processing_time_ms=_elapsed_ms(started_at),
        )

    except InvalidVideoError as error:
        logger.warning(
            "correlation_id=%s challenge_id=%s error_code=invalid_video message=%s",
            correlation_id,
            request.challenge_id,
            error,
        )
        return error_response_factory.build(
            status_code=422,
            code="invalid_video",
            message=str(error),
            correlation_id=correlation_id,
            processing_time_ms=_elapsed_ms(started_at),
        )

    except NoUsableVideoFrameError as error:
        logger.warning(
            "correlation_id=%s challenge_id=%s error_code=no_usable_video_frame message=%s",
            correlation_id,
            request.challenge_id,
            error,
        )
        return error_response_factory.build(
            status_code=422,
            code="no_usable_video_frame",
            message=str(error),
            correlation_id=correlation_id,
            processing_time_ms=_elapsed_ms(started_at),
        )

    except Exception:
        logger.exception(
            "correlation_id=%s challenge_id=%s error_code=internal_server_error",
            correlation_id,
            request.challenge_id,
        )
    return error_response_factory.build(
        status_code=500,
        code="internal_server_error",
        message="Unexpected internal server error.",
        correlation_id=correlation_id,
        processing_time_ms=_elapsed_ms(started_at),
    )


def _elapsed_ms(
    started_at: float,
) -> float:
    return round(
        (time.perf_counter() - started_at) * 1000,
        2,
    )
