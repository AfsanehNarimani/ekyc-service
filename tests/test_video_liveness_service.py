import numpy as np
from PIL import Image

from core.face_detection import FaceDetection
from core.video_frame import VideoFrame
from face.single_face_validator import SingleFaceValidator
from liveness.liveness_threshold_policy import (
    LivenessThresholdPolicy,
    LivenessThresholds,
)
from services.video_liveness_service import VideoLivenessService


class FakeFrameExtractor:
    def extract(
        self,
        video_path,
    ):
        return [
            VideoFrame(0, 0.0, Image.new("RGB", (112, 112))),
            VideoFrame(1, 1.0, Image.new("RGB", (112, 112))),
            VideoFrame(2, 2.0, Image.new("RGB", (112, 112))),
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


def test_video_liveness_service_passes_when_pose_varies(tmp_path):
    service = VideoLivenessService(
        frame_extractor=FakeFrameExtractor(),
        detector=FakeDetector(
            poses=[
                (0.0, 0.0, 0.0),
                (0.0, 25.0, 0.0),
                (0.0, -25.0, 0.0),
            ]
        ),
        validator=SingleFaceValidator(),
        threshold_policy=LivenessThresholdPolicy(
            level_to_thresholds={
                2: LivenessThresholds(
                    min_required_frames=3,
                    min_pose_variation_score=18.0,
                )
            }
        ),
    )

    result = service.check_liveness(
        video_path=tmp_path / "video.mp4",
        strictness_level=2,
    )

    assert result.passed is True
    assert result.reason == "passed_basic_liveness"
    assert result.processed_frames == 3


def test_video_liveness_service_rejects_low_pose_variation(tmp_path):
    service = VideoLivenessService(
        frame_extractor=FakeFrameExtractor(),
        detector=FakeDetector(
            poses=[
                (0.0, 1.0, 0.0),
                (0.0, 2.0, 0.0),
                (0.0, 3.0, 0.0),
            ]
        ),
        validator=SingleFaceValidator(),
        threshold_policy=LivenessThresholdPolicy(
            level_to_thresholds={
                2: LivenessThresholds(
                    min_required_frames=3,
                    min_pose_variation_score=18.0,
                )
            }
        ),
    )

    result = service.check_liveness(
        video_path=tmp_path / "video.mp4",
        strictness_level=2,
    )

    assert result.passed is False
    assert result.reason == "not_enough_pose_variation"
