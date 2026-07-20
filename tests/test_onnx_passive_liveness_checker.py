from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from core.face_detection import FaceDetection
from core.passive_liveness_result import PassiveLivenessResult
from liveness.passive_liveness_threshold_policy import (
    PassiveLivenessThresholdPolicy,
)
from services.onnx_passive_liveness_checker import ONNXPassiveLivenessChecker


class FakeFrameExtractor:
    def extract(
        self,
        video_path: Path,
    ):
        image = Image.fromarray(
            np.zeros(
                (
                    120,
                    120,
                    3,
                ),
                dtype=np.uint8,
            )
        )

        return [
            SimpleNamespace(image=image),
            SimpleNamespace(image=image),
        ]


class FakeDetector:
    def detect(
        self,
        image,
    ):
        return [
            FaceDetection(
                bbox=(
                    20,
                    20,
                    100,
                    100,
                ),
                landmarks=np.zeros(
                    (
                        5,
                        2,
                    ),
                    dtype=np.float32,
                ),
                score=0.99,
            )
        ]


class FakeValidator:
    def validate(
        self,
        detections,
    ):
        return detections[0]


class FakeModel:
    def predict_real_score(
        self,
        frame_bgr,
        detection,
    ) -> float:
        assert isinstance(frame_bgr, np.ndarray)
        assert frame_bgr.shape == (
            120,
            120,
            3,
        )

        return 0.9


def test_onnx_passive_liveness_checker_passes_when_median_score_is_above_threshold():
    checker = ONNXPassiveLivenessChecker(
        frame_extractor=FakeFrameExtractor(),
        detector=FakeDetector(),
        validator=FakeValidator(),
        model=FakeModel(),
        threshold_policy=PassiveLivenessThresholdPolicy(
            level_to_threshold={
                2: 0.6,
            }
        ),
        frame_interval_seconds=1.0,
        max_frames=20,
    )

    result = checker.check(
        video_path=Path("/tmp/video.mp4"),
        strictness_level=2,
    )

    assert isinstance(result, PassiveLivenessResult)
    assert result.passed is True
    assert result.score == 0.9
    assert result.threshold == 0.6
    assert result.model_name == "onnx_passive_liveness"
    assert result.processed_frames == 2
    assert result.skipped_frames == 0
    assert result.reason == "passed_passive_liveness"
