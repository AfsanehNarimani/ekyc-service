from dataclasses import dataclass
from datetime import datetime, timezone

from core.liveness_action import LivenessAction


@dataclass(frozen=True, slots=True)
class LivenessChallenge:
    challenge_id: str
    actions: list[LivenessAction]
    created_at: datetime
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at

    def expires_in_seconds(self) -> int:
        remaining = self.expires_at - datetime.now(timezone.utc)

        return max(
            0,
            int(remaining.total_seconds()),
        )
