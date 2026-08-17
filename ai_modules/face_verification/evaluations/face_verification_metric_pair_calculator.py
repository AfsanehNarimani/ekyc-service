from collections.abc import Sequence

from ai_modules.face_verification.evaluations.base_face_verification_metric_pair_calculator import (
    BaseFaceVerificationMetricPairCalculator,
)
from ai_modules.face_verification.evaluations.face_verification_metric_pair import (
    FaceVerificationMetricPair,
)
import math


class FaceVerificationMetricPairCalculator(BaseFaceVerificationMetricPairCalculator):
    def calculate_at_threshold(
        self,
        genuine_similarity_scores: Sequence[float],
        impostor_similarity_scores: Sequence[float],
        threshold: float,
    ) -> FaceVerificationMetricPair:
        if not genuine_similarity_scores:
            raise ValueError(
                "Genuine similarity scores must not be empty.",
            )
        if not impostor_similarity_scores:
            raise ValueError(
                "Impostor similarity scores must not be empty.",
            )

        if not math.isfinite(threshold):
            raise ValueError(
                "Threshold must be finite.",
            )

        combined_similarity_scores = (*genuine_similarity_scores, *impostor_similarity_scores)

        if not all(
            math.isfinite(similarity_score)
            for similarity_score in combined_similarity_scores
            ):
            raise ValueError(
                "Similarity scores must contain only finite values."
        )

        false_match_count = sum(
            score >= threshold for score in impostor_similarity_scores
        )

        true_match_count = sum(
            score >= threshold for score in genuine_similarity_scores
        )

        false_match_rate = false_match_count / len(impostor_similarity_scores)

        true_match_rate = true_match_count / len(genuine_similarity_scores)

        return FaceVerificationMetricPair(
            false_match_rate=false_match_rate,
            true_match_rate=true_match_rate,
        )
