import numpy as np
from PIL import Image

from core.face_detection import FaceDetection
from core.liveness_action import LivenessAction
from core.video_frame import VideoFrame
from face.single_face_validator import SingleFaceValidator
from liveness.head_pose_action_classifier import HeadPoseActionClassifier
from services.video_action_check_service import VideoActionCheckService


class FakeFrameExtractor:
    def extract(
        self,
        video_path,
    ):
        return [
            VideoFrame(0, 0.0, Image.new("RGB", (112, 112))),
            VideoFrame(1, 1.0, Image.new("RGB", (112, 112))),
            VideoFrame(2, 2.0, Image.new("RGB", (112, 112))),
            VideoFrame(3, 3.0, Image.new("RGB", (112, 112))),
            VideoFrame(4, 4.0, Image.new("RGB", (112, 112))),
            VideoFrame(5, 5.0, Image.new("RGB", (112, 112))),
        ]


class FakeDetector:
    def __init__(
        self,
        poses,
    ) -> None:
        self.poses = poses
        self.index = 0

    def detect(
        self,
        image,
    ):
        pose = self.poses[self.index]
        self.index += 1

        return [
            FaceDetection(
                bbox=(0.0, 0.0, 100.0, 100.0),
                landmarks=np.zeros((5, 2), dtype=np.float32),
                score=0.99,
                pose=pose,
            )
        ]


def make_classifier() -> HeadPoseActionClassifier:
    return HeadPoseActionClassifier(
        yaw_threshold=15.0,
        pitch_threshold=10.0,
        neutral_yaw_threshold=8.0,
        neutral_pitch_threshold=8.0,
    )


def test_video_action_check_service_passes_expected_actions(tmp_path):
    service = VideoActionCheckService(
        frame_extractor=FakeFrameExtractor(),
        detector=FakeDetector(
            poses=[
                (0.0, 0.0, 0.0),
                (0.0, 20.0, 0.0),
                (0.0, -20.0, 0.0),
                (-15.0, 0.0, 0.0),
                (15.0, 0.0, 0.0),
                (15.0, 0.0, 0.0),
            ]
        ),
        validator=SingleFaceValidator(),
        classifier=make_classifier(),
    )

    result = service.check_actions(
        video_path=tmp_path / "video.mp4",
        expected_actions=[
            LivenessAction.CENTER,
            LivenessAction.LEFT,
            LivenessAction.RIGHT,
            LivenessAction.UP,
            LivenessAction.DOWN,
        ],
    )

    assert result.passed is True
    assert result.compact_actions == [
        LivenessAction.CENTER,
        LivenessAction.LEFT,
        LivenessAction.RIGHT,
        LivenessAction.UP,
        LivenessAction.DOWN,
    ]
    assert result.missing_actions == []


def test_video_action_check_service_detects_missing_action(tmp_path):
    service = VideoActionCheckService(
        frame_extractor=FakeFrameExtractor(),
        detector=FakeDetector(
            poses=[
                (0.0, 0.0, 0.0),
                (0.0, 20.0, 0.0),
                (0.0, -20.0, 0.0),
                (15.0, 0.0, 0.0),
                (15.0, 0.0, 0.0),
                (15.0, 0.0, 0.0),
            ]
        ),
        validator=SingleFaceValidator(),
        classifier=make_classifier(),
    )

    result = service.check_actions(
        video_path=tmp_path / "video.mp4",
        expected_actions=[
            LivenessAction.CENTER,
            LivenessAction.LEFT,
            LivenessAction.RIGHT,
            LivenessAction.UP,
            LivenessAction.DOWN,
        ],
    )

    assert result.passed is False
    assert result.missing_actions == [
        LivenessAction.UP,
    ]
