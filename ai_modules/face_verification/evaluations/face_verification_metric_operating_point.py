from dataclasses import dataclass

from ai_modules.face_verification.evaluations.face_verification_metric_pair import FaceVerificationMetricPair


@dataclass(frozen=True, slots=True)
class FaceVerificationMetricOperatingPoint:
    threshold: float
    metric_pair:FaceVerificationMetricPair