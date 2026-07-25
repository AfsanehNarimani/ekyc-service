import pytest

from core.passive_liveness_calibration import (
    PassiveLivenessCalibrationEvaluator,
    PassiveLivenessCalibrationScore,
    PassiveLivenessLabel,
)


def make_samples():
    return [
        PassiveLivenessCalibrationScore(
            sample_id="real-1",
            label=PassiveLivenessLabel.BONA_FIDE,
            score=0.90,
        ),
        PassiveLivenessCalibrationScore(
            sample_id="real-2",
            label=PassiveLivenessLabel.BONA_FIDE,
            score=0.70,
        ),
        PassiveLivenessCalibrationScore(
            sample_id="attack-1",
            label=PassiveLivenessLabel.ATTACK,
            score=0.60,
        ),
        PassiveLivenessCalibrationScore(
            sample_id="attack-2",
            label=PassiveLivenessLabel.ATTACK,
            score=0.20,
        ),
    ]


def test_calibration_evaluator_calculates_pad_metrics():
    result = PassiveLivenessCalibrationEvaluator().evaluate(
        samples=make_samples(),
        thresholds=[
            0.50,
        ],
    )

    metrics = result[0]

    assert metrics.threshold == 0.50
    assert metrics.bona_fide_count == 2
    assert metrics.attack_count == 2
    assert metrics.accepted_bona_fide_count == 2
    assert metrics.rejected_bona_fide_count == 0
    assert metrics.accepted_attack_count == 1
    assert metrics.rejected_attack_count == 1
    assert metrics.apcer == 0.50
    assert metrics.bpcer == 0.0
    assert metrics.acer == 0.25


def test_calibration_evaluator_sorts_and_deduplicates_thresholds():
    result = PassiveLivenessCalibrationEvaluator().evaluate(
        samples=make_samples(),
        thresholds=[
            0.80,
            0.40,
            0.80,
        ],
    )

    assert [metrics.threshold for metrics in result] == [
        0.40,
        0.80,
    ]


def test_calibration_evaluator_requires_both_labels():
    with pytest.raises(
        ValueError,
        match="requires both",
    ):
        PassiveLivenessCalibrationEvaluator().evaluate(
            samples=[
                PassiveLivenessCalibrationScore(
                    sample_id="real-1",
                    label=(PassiveLivenessLabel.BONA_FIDE),
                    score=0.90,
                )
            ],
            thresholds=[
                0.60,
            ],
        )


@pytest.mark.parametrize(
    "threshold",
    [
        -0.01,
        1.01,
    ],
)
def test_calibration_evaluator_rejects_invalid_threshold(
    threshold,
):
    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        PassiveLivenessCalibrationEvaluator().evaluate(
            samples=make_samples(),
            thresholds=[
                threshold,
            ],
        )
