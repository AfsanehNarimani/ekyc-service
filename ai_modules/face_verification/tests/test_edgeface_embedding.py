import pytest
from PIL import Image

from ai_modules.face_verification.infrastructure.embedders.edgeface_embedder import (
    EdgeFaceEmbedder,
)
from config import settings


def test_edgeface_embedder_extracts_512d_embedding():
    if not settings.edgeface_model_path.exists():
        pytest.skip("EdgeFace ONNX model is not available locally.")

    image = Image.new(
        "RGB",
        (432, 473),
        color=(128, 128, 128),
    )

    embedder = EdgeFaceEmbedder(
        model_path=settings.edgeface_model_path,
    )

    embedding = embedder.extract(image)

    assert embedding.model_name == "edgeface"
    assert embedding.vector.shape == (512,)
    assert embedding.vector.dtype.name == "float32"
