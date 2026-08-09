from dataclasses import dataclass

from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_action_type import HeadPoseActionType


@dataclass(frozen=True, slots=True)
class ActionCheckResult:
    passed: bool
    required_actions: list[HeadPoseActionType]
    observed_actions: list[HeadPoseActionType]
    compact_actions: list[HeadPoseActionType]
    missing_actions: list[HeadPoseActionType]
    processed_frames: int
    skipped_frames: int
