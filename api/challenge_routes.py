from fastapi import APIRouter, Depends

from api.dependencies import get_challenge_service
from liveness.challenge_service import ChallengeService

router = APIRouter(
    prefix="/api",
    tags=["Liveness Challenge"],
)


@router.post("/challenges")
def create_challenge(
    service: ChallengeService = Depends(get_challenge_service),
):
    challenge = service.create_challenge()

    return {
        "ok": True,
        "challenge_id": challenge.challenge_id,
        "actions": [action.value for action in challenge.actions],
        "expires_in_seconds": challenge.expires_in_seconds(),
    }
