from pydantic import BaseModel


class FaceMatchLevelConfigResponse(BaseModel):
    name: str
    threshold: float


class LivenessLevelConfigResponse(BaseModel):
    name: str
    min_required_frames: int
    min_pose_variation_score: float


class VerificationConfigResponse(BaseModel):
    default_face_model: str
    supported_actions: list[str]
    face_match_levels: dict[str, FaceMatchLevelConfigResponse]
    liveness_levels: dict[str, LivenessLevelConfigResponse]
