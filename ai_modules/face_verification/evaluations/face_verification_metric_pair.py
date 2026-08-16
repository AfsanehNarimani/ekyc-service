from dataclasses import dataclass


@dataclass
class FaceVerificationMetricPair:
    false_match_rate: float
    false_non_match_rate: float