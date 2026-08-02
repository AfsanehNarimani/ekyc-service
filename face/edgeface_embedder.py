from pathlib import Path

import numpy as np
from PIL import Image

from ai_modules.face_verification.domain.face_embedding import FaceEmbedding
from face.base_embedder import BaseFaceEmbedder
from face.edgeface_preprocessor import EdgeFacePreprocessor
from face.onnx_runner import ONNXRunner


class EdgeFaceEmbedder(BaseFaceEmbedder):
    def __init__(
        self,
        model_path: Path,
        providers: list[str] | None = None,
    ) -> None:
        self.model_name = "edgeface"
        self.preprocessor = EdgeFacePreprocessor()
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
            model_name=self.model_name,
        )
