from core.liveness_action import LivenessAction
from core.media_reference import MediaReference
from core.video_verification_result import VideoVerificationResult
from media.image_file_loader import ImageFileLoader
from media.media_loader import MediaLoader
from services.video_action_check_service import VideoActionCheckService
from services.video_face_match_service import VideoFaceMatchService
from services.video_final_decision_service import VideoFinalDecisionService
from services.video_liveness_service import VideoLivenessService


class VideoVerificationOrchestratorService:
    def __init__(
        self,
        media_loader: MediaLoader,
        image_file_loader: ImageFileLoader,
        face_match_service: VideoFaceMatchService,
        action_check_service: VideoActionCheckService,
        liveness_service: VideoLivenessService,
        final_decision_service: VideoFinalDecisionService,
    ) -> None:
        self.media_loader = media_loader
        self.image_file_loader = image_file_loader
        self.face_match_service = face_match_service
        self.action_check_service = action_check_service
        self.liveness_service = liveness_service
        self.final_decision_service = final_decision_service

    def verify(
        self,
        challenge_id: str,
        expected_actions: list[LivenessAction],
        national_id_image_uri: str,
        video_uri: str,
        face_match_level: int,
        liveness_level: int,
    ) -> VideoVerificationResult:
        national_id_image_path = self.media_loader.load_to_file(
            reference=MediaReference(national_id_image_uri),
            suffix=".jpg",
        )
        video_path = self.media_loader.load_to_file(
            reference=MediaReference(video_uri),
            suffix=".mp4",
        )

        national_id_image = self.image_file_loader.load(
            national_id_image_path,
        )

        face_match_result = self.face_match_service.match_id_card_to_video(
            national_id_image=national_id_image,
            video_path=video_path,
            strictness_level=face_match_level,
        )

        action_check_result = self.action_check_service.check_actions(
            video_path=video_path,
            expected_actions=expected_actions,
        )

        liveness_result = self.liveness_service.check_liveness(
            video_path=video_path,
            strictness_level=liveness_level,
        )

        final_decision = self.final_decision_service.decide(
            face_match_passed=face_match_result.is_match,
            actions_passed=action_check_result.passed,
            liveness_passed=liveness_result.passed,
        )

        return VideoVerificationResult(
            challenge_id=challenge_id,
            expected_actions=expected_actions,
            face_match=face_match_result,
            action_check=action_check_result,
            liveness=liveness_result,
            final_decision=final_decision,
        )
