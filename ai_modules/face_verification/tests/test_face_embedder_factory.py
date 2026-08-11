import pytest
from PIL import Image

from ai_modules.face_verification.infrastructure.embedders.face_embedder_factory import (
    FaceEmbedderFactory,
)
from config import settings


def test_face_embedder_factory_has_all_profiles():
    factory = FaceEmbedderFactory.from_settings()

    assert factory.get_profile("edgeface").input_layout == "NCHW"
    assert factory.get_profile("ghostfacenet").input_layout == "NHWC"
    assert factory.get_profile("facenet").input_size == (160, 160)
    assert factory.get_profile("adaface").color_order == "BGR"


def test_face_embedder_factory_rejects_unknown_model():
    factory = FaceEmbedderFactory.from_settings()

    with pytest.raises(ValueError):
        factory.get_profile("unknown")


def test_face_embedder_factory_extracts_embedding_with_default_model():
    if not settings.edgeface_model_path.exists():
        pytest.skip("Default EdgeFace ONNX model is not available locally.")

    factory = FaceEmbedderFactory.from_settings()
    embedder = factory.create(settings.default_face_model)

    embedding = embedder.extract(Image.new("RGB", (300, 300), color=(128, 128, 128)))

    assert embedding.model_name == settings.default_face_model
    assert embedding.vector.shape == (512,)
