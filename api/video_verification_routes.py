from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_video_verification_orchestrator_service
from api.video_verification_schemas import (
    VerifyVideoRequest,
    VerifyVideoResponse,
)
from core.exceptions import (
    InvalidVideoError,
    MediaLoadError,
    NoUsableVideoFrameError,
)
from core.video_verification_result import VideoVerificationResult
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

        return _to_response(result)

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


def _to_response(
    result: VideoVerificationResult,
) -> VerifyVideoResponse:
    return VerifyVideoResponse(
        ok=True,
        challenge_id=result.challenge_id,
        expected_actions=[
            action.value for action in result.expected_actions
        ],
        face_match={
            "passed": result.face_match.is_match,
            "score": result.face_match.score,
            "best_score": result.face_match.best_score,
            "threshold": result.face_match.threshold,
            "model_name": result.face_match.model_name,
            "processed_frames": result.face_match.processed_frames,
            "skipped_frames": result.face_match.skipped_frames,
        },
        liveness={
            "passed": result.liveness.passed,
            "strictness_level": result.liveness.strictness_level,
            "processed_frames": result.liveness.processed_frames,
            "skipped_frames": result.liveness.skipped_frames,
            "min_required_frames": result.liveness.min_required_frames,
            "pose_variation_score": result.liveness.pose_variation_score,
            "min_pose_variation_score": (
                result.liveness.min_pose_variation_score
            ),
            "reason": result.liveness.reason,
        },
        action_check={
            "passed": result.action_check.passed,
            "required_actions": [
                action.value for action in result.action_check.required_actions
            ],
            "observed_actions": [
                action.value for action in result.action_check.observed_actions
            ],
            "compact_actions": [
                action.value for action in result.action_check.compact_actions
            ],
            "missing_actions": [
                action.value for action in result.action_check.missing_actions
            ],
            "processed_frames": result.action_check.processed_frames,
            "skipped_frames": result.action_check.skipped_frames,
        },
        final_decision={
            "approved": result.final_decision.approved,
            "status": result.final_decision.status,
            "rejection_reasons": result.final_decision.rejection_reasons,
            "face_match_passed": result.final_decision.face_match_passed,
            "actions_passed": result.final_decision.actions_passed,
            "liveness_passed": result.final_decision.liveness_passed,
        },
    )
