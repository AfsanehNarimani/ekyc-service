from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "eKYC Service"

    project_root: Path = Path(__file__).resolve().parent

    edgeface_model_path: Path = project_root / "models" / "edgeface" / "model.onnx"

    reference_dir: Path = project_root / "data" / "references"
    embedding_dir: Path = project_root / "data" / "embeddings"

    face_match_threshold: float = 0.55
    detector_threshold: float = 0.5


settings = Settings()
