from pathlib import Path

from ai_modules.face_verification.domain.face_model_name import validate_face_model_name
from ai_modules.face_verification.domain.face_model_profile import FaceModelProfile
from ai_modules.face_verification.infrastructure.embedders.onnx_face_embedder import (
    ONNXFaceEmbedder,
)
from config import settings


class FaceEmbedderFactory:
    def __init__(
        self,
        model_paths: dict[str, Path],
        providers: list[str] | None = None,
    ) -> None:
        self.model_paths = model_paths
        self.providers = providers or ["CPUExecutionProvider"]
        self.profiles = self._build_profiles()

    @classmethod
    def from_settings(
        cls,
        providers: list[str] | None = None,
    ) -> "FaceEmbedderFactory":
        return cls(
            model_paths={
                "edgeface": settings.edgeface_model_path,
                "ghostfacenet": settings.ghostfacenet_model_path,
                "mobilefacenet": settings.mobilefacenet_model_path,
                "arcface": settings.arcface_model_path,
                "facenet": settings.facenet_model_path,
                "adaface": settings.adaface_model_path,
            },
            providers=providers,
        )

    def create(
        self,
        model_name: str,
    ) -> ONNXFaceEmbedder:
        validate_face_model_name(model_name)

        return ONNXFaceEmbedder(
            model_path=self.model_paths[model_name],
            profile=self.get_profile(model_name),
            providers=self.providers,
        )

    def get_profile(
        self,
        model_name: str,
    ) -> FaceModelProfile:
        validate_face_model_name(model_name)

        return self.profiles[model_name]

    def _build_profiles(
        self,
    ) -> dict[str, FaceModelProfile]:
        return {
            "edgeface": FaceModelProfile(
                name="edgeface",
                input_size=(112, 112),
                input_layout="NCHW",
                color_order="RGB",
                normalization="minus_127_5_div_128",
            ),
            "ghostfacenet": FaceModelProfile(
                name="ghostfacenet",
                input_size=(112, 112),
                input_layout="NHWC",
                color_order="RGB",
                normalization="minus_127_5_div_128",
            ),
            "mobilefacenet": FaceModelProfile(
                name="mobilefacenet",
                input_size=(112, 112),
                input_layout="NCHW",
                color_order="RGB",
                normalization="minus_127_5_div_128",
            ),
            "arcface": FaceModelProfile(
                name="arcface",
                input_size=(112, 112),
                input_layout="NCHW",
                color_order="RGB",
                normalization="minus_127_5_div_128",
            ),
            "facenet": FaceModelProfile(
                name="facenet",
                input_size=(160, 160),
                input_layout="NCHW",
                color_order="RGB",
                normalization="minus_127_5_div_128",
            ),
            "adaface": FaceModelProfile(
                name="adaface",
                input_size=(112, 112),
                input_layout="NCHW",
                color_order="BGR",
                normalization="zero_to_one_minus_0_5_div_0_5",
            ),
        }
