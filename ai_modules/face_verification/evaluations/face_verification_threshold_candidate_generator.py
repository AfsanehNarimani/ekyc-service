import math
from typing import Sequence

from ai_modules.face_verification.evaluations.base_face_verification_threshold_candidate_generator import BaseFaceVerificationThresholdCandidateGenerator


class FaceVerificationThresholdCandidateGenerator(BaseFaceVerificationThresholdCandidateGenerator):
    def generate(
        self,
        genuine_similarity_scores: Sequence[float],
        impostor_similarity_scores: Sequence[float],
    ) -> tuple[float, ...]:
        if not genuine_similarity_scores:
            raise ValueError("Genuine similarity scores must not be empty.")

        if not impostor_similarity_scores:
            raise ValueError("Impostor similarity scores must not be empty.")

        combined_similarity_scores = (*genuine_similarity_scores,*impostor_similarity_scores)

        if not all(
            math.isfinite(similarity_score)
            for similarity_score in combined_similarity_scores
        ):
            raise ValueError("Similarity scores must contain only finite values.")

        unique_sorted_scores = sorted(
            set(combined_similarity_scores),
        )

        upper_boundary_threshold = math.nextafter(
            unique_sorted_scores[-1],
            math.inf,
        )

        return (*unique_sorted_scores, upper_boundary_threshold)