from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import (
    get_image_file_loader,
    get_media_loader,
    get_video_action_check_service,
    get_video_face_match_service,
    get_video_liveness_service,
get_video_final_decision_service,
)
from api.video_verification_schemas import VerifyVideoRequest
from core.exceptions import (
    InvalidVideoError,
    MediaLoadError,
    NoUsableVideoFrameError,
)
from core.media_reference import MediaReference
from media.image_file_loader import ImageFileLoader
from media.media_loader import MediaLoader
from services.video_action_check_service import VideoActionCheckService
from services.video_face_match_service import VideoFaceMatchService
from services.video_liveness_service import VideoLivenessService
from services.video_final_decision_service import VideoFinalDecisionService

router = APIRouter(
    prefix="/api",
    tags=["Video Verification"],
)


@router.post("/verify-video")
def verify_video(
    request: VerifyVideoRequest,
    media_loader: MediaLoader = Depends(get_media_loader),
    image_file_loader: ImageFileLoader = Depends(get_image_file_loader),
    face_match_service: VideoFaceMatchService = Depends(get_video_face_match_service),
    action_check_service: VideoActionCheckService = Depends(
        get_video_action_check_service
    ),
    liveness_service: VideoLivenessService = Depends(get_video_liveness_service),
    final_decision_service: VideoFinalDecisionService = Depends(
        get_video_final_decision_service
),
):
    try:
        national_id_image_path = media_loader.load_to_file(
            reference=MediaReference(request.national_id_image_uri),
            suffix=".jpg",
        )
        video_path = media_loader.load_to_file(
            reference=MediaReference(request.video_uri),
            suffix=".mp4",
        )

        national_id_image = image_file_loader.load(
            national_id_image_path,
        )

        face_match_result = face_match_service.match_id_card_to_video(
            national_id_image=national_id_image,
            video_path=video_path,
            strictness_level=request.face_match_level,
        )

        action_check_result = action_check_service.check_actions(
            video_path=video_path,
            expected_actions=request.expected_actions,
        )

        liveness_result = liveness_service.check_liveness(
            video_path=video_path,
            strictness_level=request.liveness_level,
        )

        final_decision = final_decision_service.decide(
            face_match_passed=face_match_result.is_match,
            actions_passed=action_check_result.passed,
            liveness_passed=liveness_result.passed,
        )

        return {
            "ok": True,
            "challenge_id": request.challenge_id,
            "expected_actions": [
                action.value for action in request.expected_actions
            ],
            "face_match": {
                "passed": face_match_result.is_match,
                "score": face_match_result.score,
                "best_score": face_match_result.best_score,
                "threshold": face_match_result.threshold,
                "model_name": face_match_result.model_name,
                "processed_frames": face_match_result.processed_frames,
                "skipped_frames": face_match_result.skipped_frames,
            },
            "liveness": {
                "passed": liveness_result.passed,
                "strictness_level": liveness_result.strictness_level,
                "processed_frames": liveness_result.processed_frames,
                "skipped_frames": liveness_result.skipped_frames,
                "min_required_frames": liveness_result.min_required_frames,
                "pose_variation_score": liveness_result.pose_variation_score,
                "min_pose_variation_score": (
                    liveness_result.min_pose_variation_score
                ),
                "reason": liveness_result.reason,
            },
            "action_check": {
                "passed": action_check_result.passed,
                "required_actions": [
                    action.value for action in action_check_result.required_actions
                ],
                "observed_actions": [
                    action.value for action in action_check_result.observed_actions
                ],
                "compact_actions": [
                    action.value for action in action_check_result.compact_actions
                ],
                "missing_actions": [
                    action.value for action in action_check_result.missing_actions
                ],
                "processed_frames": action_check_result.processed_frames,
                "skipped_frames": action_check_result.skipped_frames,
            },
            "final_decision": {
                "approved": final_decision.approved,
                "status": final_decision.status,
                "rejection_reasons": final_decision.rejection_reasons,
                "face_match_passed": final_decision.face_match_passed,
                "actions_passed": final_decision.actions_passed,
                "liveness_passed": final_decision.liveness_passed,
            },
        }

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
