from dataclasses import dataclass

from core.liveness_action import LivenessAction


@dataclass(frozen=True, slots=True)
class ObservedLivenessAction:
    action: LivenessAction
    frame_index: int
    timestamp_seconds: float
    pitch: float
    yaw: float
