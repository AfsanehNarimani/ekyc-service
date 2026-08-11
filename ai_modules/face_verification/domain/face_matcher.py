from dataclasses import dataclass

import numpy as np

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
    ) -> None:
        self.threshold = threshold

    def match(
        self,
        reference: FaceEmbedding,
        candidate: FaceEmbedding,
    ) -> FaceMatchResult:
        score = self.cosine_similarity(
            reference.vector,
            candidate.vector,
        )

        return FaceMatchResult(
            is_match=score >= self.threshold,
            score=score,
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
