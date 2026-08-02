import numpy as np
import pytest
from PIL import Image

from ai_modules.face_verification.domain.face_embedding import FaceEmbedding
from ai_modules.face_verification.domain.face_matcher import FaceMatcher
from services.image_verification_service import ImageVerificationService
from storage.embedding_store import EmbeddingStore


class FakePreprocessor:
    def preprocess(
        self,
        image: Image.Image,
    ) -> Image.Image:
        return image.resize((112, 112))


class FakeEmbedder:
    def __init__(
        self,
        vector: np.ndarray,
    ) -> None:
        self.vector = vector

    def extract(
        self,
        image: Image.Image,
    ) -> FaceEmbedding:
        return FaceEmbedding(
            vector=self.vector,
            model_name="edgeface",
        )


def test_image_verification_service_accepts_matching_face(tmp_path):
    embedding_store = EmbeddingStore(
        root_dir=tmp_path,
    )

    reference_vector = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    embedding_store.save(
        person_id="user123",
        embedding=FaceEmbedding(
            vector=reference_vector,
            model_name="edgeface",
        ),
    )

    service = ImageVerificationService(
        preprocessor=FakePreprocessor(),
        embedder=FakeEmbedder(
            vector=reference_vector,
        ),
        embedding_store=embedding_store,
        matcher=FaceMatcher(
            threshold=0.9,
        ),
    )

    result = service.verify(
        person_id="user123",
        image=Image.new("RGB", (300, 300)),
    )

    assert result.person_id == "user123"
    assert result.is_match is True
    assert result.score == 1.0
    assert result.threshold == 0.9
    assert result.model_name == "edgeface"


def test_image_verification_service_rejects_non_matching_face(tmp_path):
    embedding_store = EmbeddingStore(
        root_dir=tmp_path,
    )

    embedding_store.save(
        person_id="user123",
        embedding=FaceEmbedding(
            vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            model_name="edgeface",
        ),
    )

    service = ImageVerificationService(
        preprocessor=FakePreprocessor(),
        embedder=FakeEmbedder(
            vector=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ),
        embedding_store=embedding_store,
        matcher=FaceMatcher(
            threshold=0.9,
        ),
    )

    result = service.verify(
        person_id="user123",
        image=Image.new("RGB", (300, 300)),
    )

    assert result.is_match is False
    assert result.score == 0.0


def test_image_verification_service_rejects_missing_enrollment(tmp_path):
    service = ImageVerificationService(
        preprocessor=FakePreprocessor(),
        embedder=FakeEmbedder(
            vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        ),
        embedding_store=EmbeddingStore(
            root_dir=tmp_path,
        ),
        matcher=FaceMatcher(
            threshold=0.9,
        ),
    )

    with pytest.raises(FileNotFoundError):
        service.verify(
            person_id="user123",
            image=Image.new("RGB", (300, 300)),
        )
