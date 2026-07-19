from pathlib import Path

import numpy as np
from PIL import Image

from core.face_embedding import FaceEmbedding
from core.face_model_profile import FaceModelProfile
from face.base_embedder import BaseFaceEmbedder
from face.model_preprocessor import ModelPreprocessor
from face.onnx_runner import ONNXRunner


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
