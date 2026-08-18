# TODO should be moved to the infrastructure since it doesn't have any information about the business
from dataclasses import dataclass

import numpy as np

from ai_modules.face_verification.domain.base_face_embedding_similarity_calculator import (
    BaseFaceEmbeddingSimilarityCalculator,
)
from ai_modules.face_verification.domain.face_embedding import FaceEmbedding


@dataclass(frozen=True, slots=True)
class FaceMatchResult:
    is_match: bool
    score: float
    threshold: float


class FaceMatcher:
    def __init__(
        self,
        threshold: float,
        similarity_calculator: BaseFaceEmbeddingSimilarityCalculator,
    ) -> None:
        self.threshold = threshold
        self.similarity_calculator = similarity_calculator

    def match(
        self,
        reference: FaceEmbedding,
        candidate: FaceEmbedding,
    ) -> FaceMatchResult:
        similarity_score = self.similarity_calculator.calculate_similarity(
            reference_embedding=reference,
            candidate_embedding=candidate,
        )

        return FaceMatchResult(
            is_match=similarity_score >= self.threshold,
            score=similarity_score,
            threshold=self.threshold,
        )

    def cosine_similarity(
        self,
        vector_a: np.ndarray,
        vector_b: np.ndarray,
    ) -> float:
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(
            np.dot(vector_a, vector_b) / (norm_a * norm_b),
        )
