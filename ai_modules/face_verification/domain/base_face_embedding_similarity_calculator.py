from abc import ABC, abstractmethod

from ai_modules.face_verification.domain.face_embedding import FaceEmbedding


class BaseFaceEmbeddingSimilarityCalculator(ABC):
    @abstractmethod
    def calculate_similarity(
        self,
        reference_embedding: FaceEmbedding,
        candidate_embedding: FaceEmbedding,
    ) -> float:
        raise NotImplementedError
