import numpy as np
import pytest

from ai_modules.face_verification.domain.cosine_face_embedding_similarity_calculator import (
    CosineFaceEmbeddingSimilarityCalculator,
)
from ai_modules.face_verification.domain.face_embedding import (
    FaceEmbedding,
)


def test_calculate_cosine_similarity() -> None:
    calculator = CosineFaceEmbeddingSimilarityCalculator()

    reference_embedding = FaceEmbedding(
        vector=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        model_name="edgeface",
    )

    candidate_embedding = FaceEmbedding(
        vector=np.array(
            [1.0, 1.0],
            dtype=np.float32,
        ),
        model_name="edgeface",
    )

    similarity_score = calculator.calculate_similarity(
        reference_embedding=reference_embedding,
        candidate_embedding=candidate_embedding,
    )

    assert similarity_score == pytest.approx(
        1.0 / np.sqrt(2.0),
    )


def test_return_zero_for_zero_embedding_vector() -> None:
    calculator = CosineFaceEmbeddingSimilarityCalculator()

    reference_embedding = FaceEmbedding(
        vector=np.array(
            [0.0, 0.0],
            dtype=np.float32,
        ),
        model_name="edgeface",
    )
    candidate_embedding = FaceEmbedding(
        vector=np.array(
            [1.0, 1.0],
            dtype=np.float32,
        ),
        model_name="edgeface",
    )

    similarity_score = calculator.calculate_similarity(
        reference_embedding=reference_embedding,
        candidate_embedding=candidate_embedding,
    )

    assert similarity_score == 0.0


def test_reject_embeddings_from_different_models() -> None:
    calculator = CosineFaceEmbeddingSimilarityCalculator()

    reference_embedding = FaceEmbedding(
        vector=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        model_name="edgeface",
    )
    candidate_embedding = FaceEmbedding(
        vector=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        model_name="arcface",
    )

    with pytest.raises(
        ValueError,
        match=(r"^Face embeddings must be generated " r"by the same model\.$"),
    ):
        calculator.calculate_similarity(
            reference_embedding=reference_embedding,
            candidate_embedding=candidate_embedding,
        )


def test_reject_embedding_vectors_with_different_dimensions() -> None:
    calculator = CosineFaceEmbeddingSimilarityCalculator()

    reference_embedding = FaceEmbedding(
        vector=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        model_name="edgeface",
    )
    candidate_embedding = FaceEmbedding(
        vector=np.array(
            [1.0, 0.0, 0.5],
            dtype=np.float32,
        ),
        model_name="edgeface",
    )

    with pytest.raises(
        ValueError,
        match=(r"^Face embedding vectors must have " r"the same dimensions\.$"),
    ):
        calculator.calculate_similarity(
            reference_embedding=reference_embedding,
            candidate_embedding=candidate_embedding,
        )
