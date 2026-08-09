import numpy as np

from core.face_detection import FaceDetection
from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_action_type import HeadPoseActionType
from liveness.head_pose_action_classifier import HeadPoseActionClassifier


def make_detection(
    pitch: float,
    yaw: float,
) -> FaceDetection:
    return FaceDetection(
        bbox=(0.0, 0.0, 100.0, 100.0),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        score=0.99,
        pose=(pitch, yaw, 0.0),
    )


def make_classifier() -> HeadPoseActionClassifier:
    return HeadPoseActionClassifier(
        yaw_threshold=15.0,
        pitch_threshold=10.0,
        neutral_yaw_threshold=8.0,
        neutral_pitch_threshold=8.0,
    )


def test_classifier_detects_center():
    action = make_classifier().classify(
        make_detection(
            pitch=0.0,
            yaw=0.0,
        )
    )

    assert action == HeadPoseActionType.CENTER


def test_classifier_detects_left():
    action = make_classifier().classify(
        make_detection(
            pitch=0.0,
            yaw=20.0,
        )
    )

    assert action == HeadPoseActionType.LEFT


def test_classifier_detects_right():
    action = make_classifier().classify(
        make_detection(
            pitch=0.0,
            yaw=-20.0,
        )
    )

    assert action == HeadPoseActionType.RIGHT


def test_classifier_detects_up():
    action = make_classifier().classify(
        make_detection(
            pitch=-15.0,
            yaw=0.0,
        )
    )

    assert action == HeadPoseActionType.DOWN


def test_classifier_detects_down():
    action = make_classifier().classify(
        make_detection(
            pitch=15.0,
            yaw=0.0,
        )
    )

    assert action == HeadPoseActionType.UP
