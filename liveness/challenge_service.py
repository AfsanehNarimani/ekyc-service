from datetime import datetime, timedelta, timezone
from secrets import token_hex

from core.liveness_action import LivenessAction
from core.liveness_challenge import LivenessChallenge
from liveness.challenge_store import ChallengeStore


class ChallengeService:
    HARD_CODED_ACTIONS = [
        LivenessAction.CENTER,
        LivenessAction.LEFT,
        LivenessAction.RIGHT,
        LivenessAction.UP,
        LivenessAction.DOWN,
    ]

    def __init__(
        self,
        store: ChallengeStore,
        expires_in_seconds: int = 120,
    ) -> None:
        self.store = store
        self.expires_in_seconds = expires_in_seconds

    def create_challenge(self) -> LivenessChallenge:
        now = datetime.now(timezone.utc)

        challenge = LivenessChallenge(
            challenge_id=token_hex(16),
            actions=self.HARD_CODED_ACTIONS.copy(),
            created_at=now,
            expires_at=now
            + timedelta(
                seconds=self.expires_in_seconds,
            ),
        )

        self.store.save(challenge)

        return challenge

    def get_active_challenge(
        self,
        challenge_id: str,
    ) -> LivenessChallenge:
        challenge = self.store.get(challenge_id)

        if challenge is None:
            raise ValueError("Challenge not found.")

        if challenge.is_expired():
            self.store.delete(challenge_id)
            raise ValueError("Challenge expired.")

        return challenge
