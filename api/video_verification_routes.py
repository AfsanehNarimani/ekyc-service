from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import (
    get_image_file_loader,
    get_media_loader,
    get_video_action_check_service,
    get_video_face_match_service,
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

        final_approved = (
            face_match_result.is_match
            and action_check_result.passed
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
                "status": "pending_implementation",
                "strictness_level": request.liveness_level,
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
                "status": "pending_liveness",
                "approved": False,
                "face_match_passed": face_match_result.is_match,
                "actions_passed": action_check_result.passed,
                "liveness_passed": None,
                "would_approve_without_liveness": final_approved,
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
