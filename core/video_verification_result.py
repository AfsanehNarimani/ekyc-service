from dataclasses import dataclass

from core.action_check_result import ActionCheckResult
from core.final_decision import FinalDecision
from core.liveness_action import LivenessAction
from core.video_face_match_result import VideoFaceMatchResult
from core.video_liveness_result import VideoLivenessResult


@dataclass(frozen=True, slots=True)
class VideoVerificationResult:
    trace_id: str
    challenge_id: str
    expected_actions: list[LivenessAction]
    face_match: VideoFaceMatchResult
    action_check: ActionCheckResult
    liveness: VideoLivenessResult
    final_decision: FinalDecision
