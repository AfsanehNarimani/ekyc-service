import numpy as np

from ai_modules.face_verification.domain.cosine_face_embedding_similarity_calculator import (
    CosineFaceEmbeddingSimilarityCalculator,
)
from ai_modules.face_verification.domain.face_embedding import FaceEmbedding
from ai_modules.face_verification.domain.face_matcher import FaceMatcher


def test_face_matcher_accepts_identical_embeddings() -> None:
    vector = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    reference_embedding = FaceEmbedding(
        vector=vector,
        model_name="test",
    )
    candidate_embedding = FaceEmbedding(
        vector=vector,
        model_name="test",
    )

    matcher = FaceMatcher(
        threshold=0.9,
        similarity_calculator=CosineFaceEmbeddingSimilarityCalculator(),
    )

    result = matcher.match(
        reference=reference_embedding,
        candidate=candidate_embedding,
    )

    assert result.is_match is True
    assert result.score == 1.0
    assert result.threshold == 0.9


def test_face_matcher_rejects_different_embeddings() -> None:
    reference_embedding = FaceEmbedding(
        vector=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        model_name="test",
    )
    candidate_embedding = FaceEmbedding(
        vector=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        ),
        model_name="test",
    )

    matcher = FaceMatcher(
        threshold=0.9,
        similarity_calculator=CosineFaceEmbeddingSimilarityCalculator(),
    )

    result = matcher.match(
        reference=reference_embedding,
        candidate=candidate_embedding,
    )

    assert result.is_match is False
    assert result.score == 0.0
    assert result.threshold == 0.9
