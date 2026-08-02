import numpy as np
from PIL import Image

from ai_modules.face_verification.domain.face_embedding import FaceEmbedding
from ai_modules.face_verification.application.enrollment_service import EnrollmentService
from ai_modules.face_verification.infrastructure.persistence.embedding_store import EmbeddingStore
from ai_modules.face_verification.infrastructure.persistence.reference_image_store import ReferenceImageStore


class FakePreprocessor:
    def preprocess(
        self,
        image: Image.Image,
    ) -> Image.Image:
        return image.resize((112, 112))


class FakeEmbedder:
    def extract(
        self,
        image: Image.Image,
    ) -> FaceEmbedding:
        return FaceEmbedding(
            vector=np.ones(
                512,
                dtype=np.float32,
            ),
            model_name="edgeface",
        )


def test_enrollment_service_saves_reference_and_embedding(tmp_path):
    reference_store = ReferenceImageStore(
        root_dir=tmp_path / "references",
    )
    embedding_store = EmbeddingStore(
        root_dir=tmp_path / "embeddings",
    )

    service = EnrollmentService(
        preprocessor=FakePreprocessor(),
        embedder=FakeEmbedder(),
        reference_store=reference_store,
        embedding_store=embedding_store,
    )

    image = Image.new(
        "RGB",
        (300, 300),
        color=(128, 128, 128),
    )

    result = service.enroll(
        person_id="user123",
        image=image,
    )

    assert result.person_id == "user123"
    assert result.model_name == "edgeface"
    assert result.reference_image_path.endswith("user123.jpg")
    assert result.embedding_path.endswith("user123.npy")

    assert (tmp_path / "references" / "user123.jpg").exists()
    assert (tmp_path / "embeddings" / "user123.npy").exists()
