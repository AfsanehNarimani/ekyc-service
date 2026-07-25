from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    app_name: str = "eKYC Service"

    log_level: str = "INFO"

    keycloak_auth_enabled: bool = True
    keycloak_issuer: str | None = None
    keycloak_jwks_url: str | None = None
    keycloak_allowed_client_id: str | None = None
    keycloak_client_id_claim: str = "azp"
    keycloak_algorithm: str = "RS256"
    keycloak_jwks_cache_seconds: int = 300
    keycloak_http_timeout_seconds: float = 5.0
    keycloak_clock_skew_seconds: int = 10

    project_root: Path = Path(__file__).resolve().parent

    default_face_model: str = "edgeface"

    edgeface_model_path: Path = project_root / "models" / "edgeface" / "model.onnx"
    ghostfacenet_model_path: Path = (
        project_root / "models" / "ghostfacenet" / "model.onnx"
    )
    mobilefacenet_model_path: Path = (
        project_root / "models" / "mobilefacenet" / "model.onnx"
    )
    arcface_model_path: Path = project_root / "models" / "arcface" / "model.onnx"
    facenet_model_path: Path = project_root / "models" / "facenet" / "model.onnx"
    adaface_model_path: Path = project_root / "models" / "adaface" / "model.onnx"

    reference_dir: Path = project_root / "data" / "references"
    embedding_dir: Path = project_root / "data" / "embeddings"
    media_temp_dir: Path = project_root / "data" / "tmp" / "media"

    s3_region_name: str | None = None

    face_match_threshold: float = 0.55
    detector_threshold: float = 0.5

    insightface_model_name: str = "buffalo_l"
    insightface_root_dir: Path = Path.home() / ".insightface"
    insightface_detection_model_filename: str = "det_10g.onnx"
    insightface_pose_model_filename: str = "1k3d68.onnx"

    video_frame_interval_seconds: float = 1.0
    video_max_frames: int = 30

    max_concurrent_verifications: int = 2
    capacity_retry_after_seconds: int = 3

    face_match_threshold_level_1: float = 0.45
    face_match_threshold_level_2: float = 0.55
    face_match_threshold_level_3: float = 0.65

    head_pose_yaw_threshold: float = 15.0
    head_pose_pitch_threshold: float = 10.0
    head_pose_neutral_yaw_threshold: float = 8.0
    head_pose_neutral_pitch_threshold: float = 10.0

    liveness_min_frames_level_1: int = 3
    liveness_min_frames_level_2: int = 5
    liveness_min_frames_level_3: int = 8

    liveness_pose_variation_level_1: float = 10.0
    liveness_pose_variation_level_2: float = 18.0
    liveness_pose_variation_level_3: float = 25.0

    passive_liveness_threshold_level_1: float = 0.40
    passive_liveness_threshold_level_2: float = 0.60
    passive_liveness_threshold_level_3: float = 0.80

    passive_liveness_backend: str = "stub"
    passive_liveness_model_name: str = "minifasnet_v1se_v2_ensemble"
    passive_liveness_real_class_index: int = 1
    passive_liveness_decision_mode: str = "report_only"
    passive_liveness_crop_scale: float = 2.7
    passive_liveness_model_path: Path = (
        project_root / "models" / "passive_liveness" / "minifasnet_v2.onnx"
    )
    passive_liveness_input_width: int = 80
    passive_liveness_input_height: int = 80
    passive_liveness_frame_interval_seconds: float = 1.0
    passive_liveness_max_frames: int = 20

    passive_liveness_pixel_divisor: float = 1.0

    passive_liveness_v1se_crop_scale: float = 4.0
    passive_liveness_v1se_model_path: Path = (
        project_root / "models" / "passive_liveness" / "minifasnet_v1se.onnx"
    )


settings = Settings()
