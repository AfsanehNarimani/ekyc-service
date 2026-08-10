import numpy as np
from PIL import Image

from core.analyzed_video_frame import (
    AnalyzedVideoFrame,
    VideoFrameAnalysisResult,
)
from core.face_detection import FaceDetection
from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_action_type import HeadPoseActionType
from core.video_frame import VideoFrame
from ai_modules.challenge_based_liveness_detection.head_pose_estimation \
    .infrastructure.head_pose_action_detector import HeadPoseActionDetector
from ai_modules.challenge_based_liveness_detection.head_pose_estimation \
    .application.head_pose_challenge_check_service import HeadPoseChallengeCheckService


def make_head_pose_action_detector() -> HeadPoseActionDetector:
    return HeadPoseActionDetector(
        yaw_threshold=15.0,
        pitch_threshold=10.0,
        neutral_yaw_threshold=8.0,
        neutral_pitch_threshold=8.0,
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


def test_video_action_check_service_passes_expected_actions():
    service = HeadPoseChallengeCheckService(
        head_pose_action_detector=make_head_pose_action_detector(),
    )

    result = service.check_head_pose_actions(
        video_analysis=make_video_analysis(
            poses=[
                (0.0, 0.0, 0.0),
                (0.0, 20.0, 0.0),
                (0.0, -20.0, 0.0),
                (-15.0, 0.0, 0.0),
                (15.0, 0.0, 0.0),
                (15.0, 0.0, 0.0),
            ]
        ),
        expected_actions=[
            HeadPoseActionType.CENTER,
            HeadPoseActionType.LEFT,
            HeadPoseActionType.RIGHT,
            HeadPoseActionType.UP,
            HeadPoseActionType.DOWN,
        ],
    )

    assert result.passed is True
    assert result.compact_actions == [
        HeadPoseActionType.CENTER,
        HeadPoseActionType.LEFT,
        HeadPoseActionType.RIGHT,
        HeadPoseActionType.DOWN,
        HeadPoseActionType.UP,
    ]
    assert result.missing_actions == []


def test_video_action_check_service_detects_missing_action():
    service = HeadPoseChallengeCheckService(
        head_pose_action_detector=make_head_pose_action_detector(),
    )

    result = service.check_head_pose_actions(
        video_analysis=make_video_analysis(
            poses=[
                (0.0, 0.0, 0.0),
                (0.0, 20.0, 0.0),
                (0.0, -20.0, 0.0),
                (15.0, 0.0, 0.0),
                (15.0, 0.0, 0.0),
                (15.0, 0.0, 0.0),
            ]
        ),
        expected_actions=[
            HeadPoseActionType.CENTER,
            HeadPoseActionType.LEFT,
            HeadPoseActionType.RIGHT,
            HeadPoseActionType.UP,
            HeadPoseActionType.DOWN,
        ],
    )

    assert result.passed is False
    assert result.missing_actions == [
        HeadPoseActionType.DOWN,
    ]
