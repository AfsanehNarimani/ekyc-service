from collections.abc import Sequence

from ai_modules.face_verification.evaluations.base_face_verification_metric_pair_calculator import \
    BaseFaceVerificationMetricPairCalculator
from ai_modules.face_verification.evaluations.face_verification_metric_pair import FaceVerificationMetricPair


class FaceVerificationMetricPairCalculator(BaseFaceVerificationMetricPairCalculator):
    def calculate_at_threshold(
        self,
        genuine_similarity_scores: Sequence[float],
        imposter_similarity_scores: Sequence[float],
        threshold: float,
    ) -> FaceVerificationMetricPair:
        false_match_count = sum(
        score>=threshold
            for score in imposter_similarity_scores
        )

        false_non_match_count = sum(
            score < threshold
            for score in genuine_similarity_scores
        )

        false_match_rate = (false_match_count / len(imposter_similarity_scores))

        false_non_match_rate = (false_non_match_count / len(genuine_similarity_scores))

        return FaceVerificationMetricPair(
            false_match_rate = false_match_rate,
            false_non_match_rate = false_non_match_rate,
        )

