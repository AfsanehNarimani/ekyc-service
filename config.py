from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "eKYC Service"

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

    video_frame_interval_seconds: float = 1.0
    video_max_frames: int = 30

    face_match_threshold_level_1: float = 0.45
    face_match_threshold_level_2: float = 0.55
    face_match_threshold_level_3: float = 0.65

    head_pose_yaw_threshold: float = 15.0
    head_pose_pitch_threshold: float = 10.0
    head_pose_neutral_yaw_threshold: float = 8.0
    head_pose_neutral_pitch_threshold: float = 8.0


settings = Settings()
