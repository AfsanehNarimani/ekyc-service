from core.liveness_challenge import LivenessChallenge


class ChallengeStore:
    def __init__(self) -> None:
        self._challenges: dict[str, LivenessChallenge] = {}

    def save(
        self,
        challenge: LivenessChallenge,
    ) -> None:
        self._challenges[challenge.challenge_id] = challenge

    def get(
        self,
        challenge_id: str,
    ) -> LivenessChallenge | None:
        return self._challenges.get(challenge_id)

    def delete(
        self,
        challenge_id: str,
    ) -> None:
        self._challenges.pop(
            challenge_id,
            None,
        )
