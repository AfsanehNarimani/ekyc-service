from pydantic import BaseModel, Field

from core.liveness_action import LivenessAction


class VerifyVideoRequest(BaseModel):
    correlation_id: str | None = Field(default=None, min_length=1)
    challenge_id: str | None = Field(default=None, min_length=1)
    expected_actions: list[LivenessAction] = Field(min_length=1)
    national_id_image_uri: str = Field(min_length=1)
    video_uri: str = Field(min_length=1)
    face_match_level: int = Field(ge=1, le=3)
    liveness_level: int = Field(ge=1, le=3)


class FaceMatchResponse(BaseModel):
    passed: bool
    score: float
    best_score: float
    threshold: float
    model_name: str
    processed_frames: int
    skipped_frames: int


class LivenessResponse(BaseModel):
    passed: bool
    strictness_level: int
    processed_frames: int
    skipped_frames: int
    min_required_frames: int
    pose_variation_score: float
    min_pose_variation_score: float
    reason: str


class PassiveLivenessResponse(BaseModel):
    passed: bool
    score: float
    threshold: float
    model_name: str
    processed_frames: int
    skipped_frames: int
    reason: str


class ActionCheckResponse(BaseModel):
    passed: bool
    required_actions: list[str]
    observed_actions: list[str]
    compact_actions: list[str]
    missing_actions: list[str]
    processed_frames: int
    skipped_frames: int


class FinalDecisionResponse(BaseModel):
    approved: bool
    status: str
    rejection_reasons: list[str]
    face_match_passed: bool
    actions_passed: bool
    liveness_passed: bool
    passive_liveness_passed: bool
    passive_liveness_enforced: bool


class VerifyVideoResponse(BaseModel):
    ok: bool
    correlation_id: str
    challenge_id: str
    expected_actions: list[str]
    face_match: FaceMatchResponse
    liveness: LivenessResponse
    passive_liveness: PassiveLivenessResponse
    action_check: ActionCheckResponse
    final_decision: FinalDecisionResponse
    processing_time_ms: float
