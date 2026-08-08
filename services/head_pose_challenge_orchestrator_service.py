import logging
import time
from pathlib import Path
from config import settings

from core.liveness_action import LivenessAction
from core.media_reference import MediaReference
from core.video_verification_result import VideoVerificationResult
from liveness.base_passive_liveness_checker import BasePassiveLivenessChecker
from liveness.expected_action_policy import ExpectedActionPolicy
from media.image_file_loader import ImageFileLoader
from media.media_cleanup_service import MediaCleanupService
from media.media_loader import MediaLoader
from services.video_action_check_service import VideoActionCheckService
from ai_modules.face_verification.application.video_face_match_service import VideoFaceMatchService
from services.video_final_decision_service import VideoFinalDecisionService
from services.video_liveness_service import VideoLivenessService
from services.video_frame_analysis_service import (
    VideoFrameAnalysisService,
)

logger = logging.getLogger(__name__)


class HeadPoseChallengeOrchestratorService:
    def __init__(
        self,
        media_loader: MediaLoader,
        image_file_loader: ImageFileLoader,
        video_frame_analysis_service: VideoFrameAnalysisService,
        face_match_service: VideoFaceMatchService,
        action_check_service: VideoActionCheckService,
        liveness_service: VideoLivenessService,
        passive_liveness_checker: BasePassiveLivenessChecker,
        final_decision_service: VideoFinalDecisionService,
        expected_action_policy: ExpectedActionPolicy,
        media_cleanup_service: MediaCleanupService,
    ) -> None:
        self.media_loader = media_loader
        self.image_file_loader = image_file_loader
        self.video_frame_analysis_service = video_frame_analysis_service
        self.face_match_service = face_match_service
        self.action_check_service = action_check_service
        self.liveness_service = liveness_service
        self.passive_liveness_checker = passive_liveness_checker
        self.final_decision_service = final_decision_service
        self.expected_action_policy = expected_action_policy
        self.media_cleanup_service = media_cleanup_service

    def verify(
        self,
        correlation_id: str,
        challenge_id: str,
        expected_actions: list[LivenessAction],
        national_id_image_uri: str,
        video_uri: str,
        face_match_level: int,
        liveness_level: int,
    ) -> VideoVerificationResult:
        started_at = time.perf_counter()

        logger.info(
            "correlation_id=%s challenge_id=%s step=verification_started",
            correlation_id,
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

            video_analysis = self.video_frame_analysis_service.analyze(
                video_path=video_path,
            )

            face_match_result = self.face_match_service.match_id_card_to_video(
                national_id_image=national_id_image,
                video_analysis=video_analysis,
                strictness_level=face_match_level,
            )

            logger.info(
                "correlation_id=%s challenge_id=%s step=face_match_done passed=%s score=%.4f",
                correlation_id,
                challenge_id,
                face_match_result.is_match,
                face_match_result.score,
            )

            action_check_result = self.action_check_service.check_actions(
                video_analysis=video_analysis,
                expected_actions=expected_actions,
            )

            logger.info(
                "correlation_id=%s challenge_id=%s step=action_check_done passed=%s missing=%s",
                correlation_id,
                challenge_id,
                action_check_result.passed,
                [action.value for action in action_check_result.missing_actions],
            )

            liveness_result = self.liveness_service.check_liveness(
                video_analysis=video_analysis,
                strictness_level=liveness_level,
            )

            logger.info(
                "correlation_id=%s challenge_id=%s step=liveness_done passed=%s reason=%s",
                correlation_id,
                challenge_id,
                liveness_result.passed,
                liveness_result.reason,
            )

            passive_liveness_result = self.passive_liveness_checker.check(
                video_analysis=video_analysis,
                strictness_level=liveness_level,
            )

            logger.info(
                "correlation_id=%s challenge_id=%s step=passive_liveness_done passed=%s score=%.4f reason=%s",
                correlation_id,
                challenge_id,
                passive_liveness_result.passed,
                passive_liveness_result.score,
                passive_liveness_result.reason,
            )

            passive_liveness_enforced = (
                settings.passive_liveness_decision_mode == "enforce"
            )

            final_decision = self.final_decision_service.decide(
                face_match_passed=face_match_result.is_match,
                actions_passed=action_check_result.passed,
                liveness_passed=liveness_result.passed,
                passive_liveness_passed=passive_liveness_result.passed,
                passive_liveness_enforced=passive_liveness_enforced,
            )

            processing_time_ms = round(
                (time.perf_counter() - started_at) * 1000,
                2,
            )

            logger.info(
                "correlation_id=%s challenge_id=%s step=verification_finished approved=%s reasons=%s processing_time_ms=%.2f",
                correlation_id,
                challenge_id,
                final_decision.approved,
                final_decision.rejection_reasons,
                processing_time_ms,
            )

            return VideoVerificationResult(
                correlation_id=correlation_id,
                challenge_id=challenge_id,
                expected_actions=expected_actions,
                face_match=face_match_result,
                action_check=action_check_result,
                liveness=liveness_result,
                passive_liveness=passive_liveness_result,
                final_decision=final_decision,
                processing_time_ms=processing_time_ms,
            )

        finally:
            self.media_cleanup_service.cleanup_temp_files(
                paths=temp_candidates,
                correlation_id=correlation_id,
            )
