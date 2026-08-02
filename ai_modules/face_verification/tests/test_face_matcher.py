import numpy as np

from ai_modules.face_verification.domain.face_embedding import FaceEmbedding
from ai_modules.face_verification.domain.face_matcher import FaceMatcher


def test_face_matcher_accepts_identical_embeddings():
    vector = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    reference = FaceEmbedding(
        vector=vector,
        model_name="test",
    )
    candidate = FaceEmbedding(
        vector=vector,
        model_name="test",
    )

    result = FaceMatcher(
        threshold=0.9,
    ).match(
        reference=reference,
        candidate=candidate,
    )

    assert result.is_match is True
    assert result.score == 1.0


def test_face_matcher_rejects_different_embeddings():
    reference = FaceEmbedding(
        vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        model_name="test",
    )
    candidate = FaceEmbedding(
        vector=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        model_name="test",
    )

    result = FaceMatcher(
        threshold=0.9,
    ).match(
        reference=reference,
        candidate=candidate,
    )

    assert result.is_match is False
    assert result.score == 0.0
