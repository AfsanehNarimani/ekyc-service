from dataclasses import dataclass


@dataclass
class FaceVerificationMetricPair:
    false_match_rate: float
    true_match_rate: float
