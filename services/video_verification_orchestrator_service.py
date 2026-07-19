import logging
from pathlib import Path

from core.liveness_action import LivenessAction
from core.media_reference import MediaReference
from core.video_verification_result import VideoVerificationResult
from liveness.expected_action_policy import ExpectedActionPolicy
from media.image_file_loader import ImageFileLoader
from media.media_cleanup_service import MediaCleanupService
from media.media_loader import MediaLoader
from services.video_action_check_service import VideoActionCheckService
from services.video_face_match_service import VideoFaceMatchService
from services.video_final_decision_service import VideoFinalDecisionService
from services.video_liveness_service import VideoLivenessService

logger = logging.getLogger(__name__)


class VideoVerificationOrchestratorService:
    def __init__(
        self,
        media_loader: MediaLoader,
        image_file_loader: ImageFileLoader,
        face_match_service: VideoFaceMatchService,
        action_check_service: VideoActionCheckService,
        liveness_service: VideoLivenessService,
        final_decision_service: VideoFinalDecisionService,
        expected_action_policy: ExpectedActionPolicy,
        media_cleanup_service: MediaCleanupService,
    ) -> None:
        self.media_loader = media_loader
        self.image_file_loader = image_file_loader
        self.face_match_service = face_match_service
        self.action_check_service = action_check_service
        self.liveness_service = liveness_service
        self.final_decision_service = final_decision_service
        self.expected_action_policy = expected_action_policy
        self.media_cleanup_service = media_cleanup_service

    def verify(
        self,
        trace_id: str,
        challenge_id: str,
        expected_actions: list[LivenessAction],
        national_id_image_uri: str,
        video_uri: str,
        face_match_level: int,
        liveness_level: int,
    ) -> VideoVerificationResult:
        logger.info(
            "trace_id=%s challenge_id=%s step=verification_started",
            trace_id,
            challenge_id,
        )

        self.expected_action_policy.validate(expected_actions)

        temp_candidates: list[Path] = []

        try:
            national_id_image_path = self.media_loader.load_to_file(
                reference=MediaReference(national_id_image_uri),
                suffix=".jpg",
            )
            temp_candidates.append(national_id_image_path)

            video_path = self.media_loader.load_to_file(
                reference=MediaReference(video_uri),
                suffix=".mp4",
            )
            temp_candidates.append(video_path)

            national_id_image = self.image_file_loader.load(
                national_id_image_path,
            )

            face_match_result = self.face_match_service.match_id_card_to_video(
                national_id_image=national_id_image,
                video_path=video_path,
                strictness_level=face_match_level,
            )

            logger.info(
                "trace_id=%s challenge_id=%s step=face_match_done passed=%s score=%.4f",
                trace_id,
                challenge_id,
                face_match_result.is_match,
                face_match_result.score,
            )

            action_check_result = self.action_check_service.check_actions(
                video_path=video_path,
                expected_actions=expected_actions,
            )

            logger.info(
                "trace_id=%s challenge_id=%s step=action_check_done passed=%s missing=%s",
                trace_id,
                challenge_id,
                action_check_result.passed,
                [action.value for action in action_check_result.missing_actions],
            )

            liveness_result = self.liveness_service.check_liveness(
                video_path=video_path,
                strictness_level=liveness_level,
            )

            logger.info(
                "trace_id=%s challenge_id=%s step=liveness_done passed=%s reason=%s",
                trace_id,
                challenge_id,
                liveness_result.passed,
                liveness_result.reason,
            )

            final_decision = self.final_decision_service.decide(
                face_match_passed=face_match_result.is_match,
                actions_passed=action_check_result.passed,
                liveness_passed=liveness_result.passed,
            )

            logger.info(
                "trace_id=%s challenge_id=%s step=verification_finished approved=%s reasons=%s",
                trace_id,
                challenge_id,
                final_decision.approved,
                final_decision.rejection_reasons,
            )

            return VideoVerificationResult(
                trace_id=trace_id,
                challenge_id=challenge_id,
                expected_actions=expected_actions,
                face_match=face_match_result,
                action_check=action_check_result,
                liveness=liveness_result,
                final_decision=final_decision,
            )

        finally:
            self.media_cleanup_service.cleanup_temp_files(
                paths=temp_candidates,
                trace_id=trace_id,
            )
