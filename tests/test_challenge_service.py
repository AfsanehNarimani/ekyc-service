import pytest

from core.liveness_action import LivenessAction
from liveness.challenge_service import ChallengeService
from liveness.challenge_store import ChallengeStore


def test_challenge_service_creates_hardcoded_challenge():
    service = ChallengeService(
        store=ChallengeStore(),
        expires_in_seconds=120,
    )

    challenge = service.create_challenge()

    assert challenge.challenge_id
    assert challenge.actions == [
        LivenessAction.CENTER,
        LivenessAction.LEFT,
        LivenessAction.RIGHT,
        LivenessAction.UP,
        LivenessAction.DOWN,
    ]
    assert challenge.expires_in_seconds() <= 120
    assert challenge.expires_in_seconds() > 0


def test_challenge_service_returns_active_challenge():
    service = ChallengeService(
        store=ChallengeStore(),
        expires_in_seconds=120,
    )

    challenge = service.create_challenge()

    active_challenge = service.get_active_challenge(
        challenge.challenge_id,
    )

    assert active_challenge.challenge_id == challenge.challenge_id


def test_challenge_service_rejects_unknown_challenge():
    service = ChallengeService(
        store=ChallengeStore(),
        expires_in_seconds=120,
    )

    with pytest.raises(ValueError, match="Challenge not found"):
        service.get_active_challenge("missing")
