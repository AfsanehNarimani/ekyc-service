from pydantic import BaseModel, Field

from core.liveness_action import LivenessAction


class VerifyVideoRequest(BaseModel):
    challenge_id: str = Field(
        min_length=1,
    )
    expected_actions: list[LivenessAction] = Field(
        min_length=1,
    )
    national_id_image_uri: str = Field(
        min_length=1,
    )
    video_uri: str = Field(
        min_length=1,
    )
    face_match_level: int = Field(
        ge=1,
        le=3,
    )
    liveness_level: int = Field(
        ge=1,
        le=3,
    )
