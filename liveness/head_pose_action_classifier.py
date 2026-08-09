from core.face_detection import FaceDetection
from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_action import HeadPoseAction


class HeadPoseActionClassifier:
    def __init__(
        self,
        yaw_threshold: float,
        pitch_threshold: float,
        neutral_yaw_threshold: float,
        neutral_pitch_threshold: float,
    ) -> None:
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.neutral_yaw_threshold = neutral_yaw_threshold
        self.neutral_pitch_threshold = neutral_pitch_threshold

    def classify(
        self,
        detection: FaceDetection,
    ) -> HeadPoseAction | None:
        if detection.pose is None:
            return None

        pitch, yaw, _roll = detection.pose

        if (
            abs(yaw) <= self.neutral_yaw_threshold
            and abs(pitch) <= self.neutral_pitch_threshold
        ):
            return HeadPoseAction.CENTER

        if yaw >= self.yaw_threshold:
            return HeadPoseAction.LEFT

        if yaw <= -self.yaw_threshold:
            return HeadPoseAction.RIGHT

        if pitch >= self.pitch_threshold:
            return HeadPoseAction.UP

        if pitch <= -self.pitch_threshold:
            return HeadPoseAction.DOWN

        return None
