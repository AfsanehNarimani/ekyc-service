from pydantic import BaseModel


class FaceMatchLevelConfigResponse(BaseModel):
    name: str
    threshold: float


class LivenessLevelConfigResponse(BaseModel):
    name: str
    min_required_frames: int
    min_pose_variation_score: float


class PassiveLivenessLevelConfigResponse(BaseModel):
    name: str
    threshold: float


class VerificationConfigResponse(BaseModel):
    default_face_model: str
    supported_actions: list[str]
    face_match_levels: dict[str, FaceMatchLevelConfigResponse]
    liveness_levels: dict[str, LivenessLevelConfigResponse]
    passive_liveness_levels: dict[str, PassiveLivenessLevelConfigResponse]
