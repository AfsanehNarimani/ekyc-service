from dataclasses import dataclass

from core.liveness_action import LivenessAction


@dataclass(frozen=True, slots=True)
class ActionCheckResult:
    passed: bool
    required_actions: list[LivenessAction]
    observed_actions: list[LivenessAction]
    compact_actions: list[LivenessAction]
    missing_actions: list[LivenessAction]
    processed_frames: int
    skipped_frames: int
