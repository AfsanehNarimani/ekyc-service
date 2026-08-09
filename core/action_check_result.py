from dataclasses import dataclass

from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_action import HeadPoseAction


@dataclass(frozen=True, slots=True)
class ActionCheckResult:
    passed: bool
    required_actions: list[HeadPoseAction]
    observed_actions: list[HeadPoseAction]
    compact_actions: list[HeadPoseAction]
    missing_actions: list[HeadPoseAction]
    processed_frames: int
    skipped_frames: int
