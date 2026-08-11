from pathlib import Path

import numpy as np
from PIL import Image

from ai_modules.face_verification.domain.face_embedding import FaceEmbedding
from ai_modules.face_verification.domain.face_model_profile import FaceModelProfile
from ai_modules.face_verification.infrastructure.embedders.base_embedder import (
    BaseFaceEmbedder,
)
from ai_modules.face_verification.infrastructure.inference.onnx_runner import ONNXRunner
from ai_modules.face_verification.infrastructure.preprocessing.model_preprocessor import (
    ModelPreprocessor,
)


class ONNXFaceEmbedder(BaseFaceEmbedder):
    def __init__(
        self,
        model_path: Path,
        profile: FaceModelProfile,
        providers: list[str] | None = None,
    ) -> None:
        self.profile = profile
        self.preprocessor = ModelPreprocessor(
            profile=profile,
        )
        self.runner = ONNXRunner(
            model_path=model_path,
            providers=providers,
        )

    def extract(
        self,
        image: Image.Image,
    ) -> FaceEmbedding:
        input_tensor = self.preprocessor.preprocess(image)
        output = self.runner.run(input_tensor)

        vector = output.squeeze().astype(np.float32)

        return FaceEmbedding(
            vector=vector,
            model_name=self.profile.name,
        )
