from dataclasses import dataclass

from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_challenge_check_result import HeadPoseChallengeCheckResult
from core.final_decision import FinalDecision
from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_action_type import HeadPoseActionType
from core.passive_liveness_result import PassiveLivenessResult
from ai_modules.face_verification.domain.video_face_match_result import VideoFaceMatchResult
from core.video_liveness_result import VideoLivenessResult


@dataclass(frozen=True, slots=True)
class VideoVerificationResult:
    correlation_id: str
    challenge_id: str
    expected_actions: list[HeadPoseActionType]
    face_match: VideoFaceMatchResult
    action_check: HeadPoseChallengeCheckResult
    liveness: VideoLivenessResult
    passive_liveness: PassiveLivenessResult
    final_decision: FinalDecision
    processing_time_ms: float
