import numpy as np
from PIL import Image

from core.analyzed_video_frame import (
    AnalyzedVideoFrame,
    VideoFrameAnalysisResult,
)
from core.face_detection import FaceDetection
from core.passive_liveness_result import PassiveLivenessResult
from core.video_frame import VideoFrame
from liveness.passive_liveness_threshold_policy import (
    PassiveLivenessThresholdPolicy,
)
from services.onnx_passive_liveness_checker import (
    ONNXPassiveLivenessChecker,
)


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


def make_video_analysis() -> VideoFrameAnalysisResult:
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

    detection = FaceDetection(
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

    return VideoFrameAnalysisResult(
        frames=(
            AnalyzedVideoFrame(
                frame=VideoFrame(
                    frame_index=0,
                    timestamp_seconds=0.0,
                    image=image,
                ),
                detection=detection,
            ),
            AnalyzedVideoFrame(
                frame=VideoFrame(
                    frame_index=25,
                    timestamp_seconds=1.0,
                    image=image,
                ),
                detection=detection,
            ),
        ),
        total_frames=2,
        skipped_frames=0,
    )


def test_onnx_passive_liveness_checker_passes_when_median_score_is_above_threshold():
    checker = ONNXPassiveLivenessChecker(
        model=FakeModel(),
        threshold_policy=PassiveLivenessThresholdPolicy(
            level_to_threshold={
                2: 0.6,
            }
        ),
    )

    result = checker.check(
        video_analysis=make_video_analysis(),
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
