from dataclasses import dataclass

from core.action_check_result import ActionCheckResult
from core.final_decision import FinalDecision
from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_action import HeadPoseAction
from core.passive_liveness_result import PassiveLivenessResult
from ai_modules.face_verification.domain.video_face_match_result import VideoFaceMatchResult
from core.video_liveness_result import VideoLivenessResult


@dataclass(frozen=True, slots=True)
class VideoVerificationResult:
    correlation_id: str
    challenge_id: str
    expected_actions: list[HeadPoseAction]
    face_match: VideoFaceMatchResult
    action_check: ActionCheckResult
    liveness: VideoLivenessResult
    passive_liveness: PassiveLivenessResult
    final_decision: FinalDecision
    processing_time_ms: float
