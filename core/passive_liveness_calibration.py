from dataclasses import dataclass
from enum import Enum


class PassiveLivenessLabel(str, Enum):
    BONA_FIDE = "bona_fide"
    ATTACK = "attack"


@dataclass(frozen=True, slots=True)
class PassiveLivenessCalibrationScore:
    sample_id: str
    label: PassiveLivenessLabel
    score: float

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty.")

        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class PassiveLivenessThresholdMetrics:
    threshold: float
    bona_fide_count: int
    attack_count: int
    accepted_bona_fide_count: int
    rejected_bona_fide_count: int
    accepted_attack_count: int
    rejected_attack_count: int
    apcer: float
    bpcer: float
    acer: float


class PassiveLivenessCalibrationEvaluator:
    def evaluate(
        self,
        samples: list[PassiveLivenessCalibrationScore],
        thresholds: list[float],
    ) -> list[PassiveLivenessThresholdMetrics]:
        self._validate_samples(samples)
        self._validate_thresholds(thresholds)

        return [
            self._evaluate_threshold(
                samples=samples,
                threshold=threshold,
            )
            for threshold in sorted(set(thresholds))
        ]

    def _evaluate_threshold(
        self,
        samples: list[PassiveLivenessCalibrationScore],
        threshold: float,
    ) -> PassiveLivenessThresholdMetrics:
        bona_fide_samples = [
            sample
            for sample in samples
            if sample.label is PassiveLivenessLabel.BONA_FIDE
        ]
        attack_samples = [
            sample for sample in samples if sample.label is PassiveLivenessLabel.ATTACK
        ]

        accepted_bona_fide_count = sum(
            sample.score >= threshold for sample in bona_fide_samples
        )
        accepted_attack_count = sum(
            sample.score >= threshold for sample in attack_samples
        )

        rejected_bona_fide_count = len(bona_fide_samples) - accepted_bona_fide_count
        rejected_attack_count = len(attack_samples) - accepted_attack_count

        apcer = accepted_attack_count / len(attack_samples)
        bpcer = rejected_bona_fide_count / len(bona_fide_samples)
        acer = (apcer + bpcer) / 2.0

        return PassiveLivenessThresholdMetrics(
            threshold=threshold,
            bona_fide_count=len(bona_fide_samples),
            attack_count=len(attack_samples),
            accepted_bona_fide_count=(accepted_bona_fide_count),
            rejected_bona_fide_count=(rejected_bona_fide_count),
            accepted_attack_count=(accepted_attack_count),
            rejected_attack_count=(rejected_attack_count),
            apcer=apcer,
            bpcer=bpcer,
            acer=acer,
        )

    def _validate_samples(
        self,
        samples: list[PassiveLivenessCalibrationScore],
    ) -> None:
        if not samples:
            raise ValueError("calibration samples must not be empty.")

        labels = {sample.label for sample in samples}

        required_labels = {
            PassiveLivenessLabel.BONA_FIDE,
            PassiveLivenessLabel.ATTACK,
        }

        if labels != required_labels:
            raise ValueError(
                "calibration requires both bona_fide " "and attack samples."
            )

    def _validate_thresholds(
        self,
        thresholds: list[float],
    ) -> None:
        if not thresholds:
            raise ValueError("thresholds must not be empty.")

        if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
            raise ValueError("thresholds must be between 0.0 and 1.0.")
