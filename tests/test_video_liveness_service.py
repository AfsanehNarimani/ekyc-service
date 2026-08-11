import numpy as np
from PIL import Image

from ai_modules.face_detection.domain.analyzed_video_frame import (
    AnalyzedVideoFrame,
    VideoFrameAnalysisResult,
)
from ai_modules.face_detection.domain.face_detection import FaceDetection
from core.video_frame import VideoFrame
from liveness.liveness_threshold_policy import (
    LivenessThresholdPolicy,
    LivenessThresholds,
)
from services.video_liveness_service import (
    VideoLivenessService,
)


def make_video_analysis(
    poses: list[tuple[float, float, float]],
) -> VideoFrameAnalysisResult:
    analyzed_frames = []

    for index, pose in enumerate(poses):
        analyzed_frames.append(
            AnalyzedVideoFrame(
                frame=VideoFrame(
                    frame_index=index,
                    timestamp_seconds=float(index),
                    image=Image.new(
                        "RGB",
                        (
                            112,
                            112,
                        ),
                    ),
                ),
                detection=FaceDetection(
                    bbox=(
                        0.0,
                        0.0,
                        100.0,
                        100.0,
                    ),
                    landmarks=np.zeros(
                        (
                            5,
                            2,
                        ),
                        dtype=np.float32,
                    ),
                    score=0.99,
                    pose=pose,
                ),
            )
        )

    return VideoFrameAnalysisResult(
        frames=tuple(analyzed_frames),
        total_frames=len(analyzed_frames),
        skipped_frames=0,
    )


def make_service() -> VideoLivenessService:
    return VideoLivenessService(
        threshold_policy=LivenessThresholdPolicy(
            level_to_thresholds={
                2: LivenessThresholds(
                    min_required_frames=3,
                    min_pose_variation_score=18.0,
                )
            }
        ),
    )


def test_video_liveness_service_passes_when_pose_varies():
    result = make_service().check_liveness(
        video_analysis=make_video_analysis(
            poses=[
                (0.0, 0.0, 0.0),
                (0.0, 25.0, 0.0),
                (0.0, -25.0, 0.0),
            ]
        ),
        strictness_level=2,
    )

    assert result.passed is True
    assert result.reason == "passed_basic_liveness"
    assert result.processed_frames == 3


def test_video_liveness_service_rejects_low_pose_variation():
    result = make_service().check_liveness(
        video_analysis=make_video_analysis(
            poses=[
                (0.0, 1.0, 0.0),
                (0.0, 2.0, 0.0),
                (0.0, 3.0, 0.0),
            ]
        ),
        strictness_level=2,
    )

    assert result.passed is False
    assert result.reason == "not_enough_pose_variation"
