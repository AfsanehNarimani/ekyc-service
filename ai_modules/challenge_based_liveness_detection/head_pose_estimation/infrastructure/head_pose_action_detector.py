from typing import TypeAlias

from ai_modules.challenge_based_liveness_detection.head_pose_estimation. \
    domain.head_pose_action_type import HeadPoseActionType
from ai_modules.challenge_based_liveness_detection.head_pose_estimation. \
    infrastructure.head_pose_action_classifier import HeadPoseActionClassifier
# TODO should be refactored
from ai_modules.face_detection.domain.analyzed_video_frame import VideoFrameAnalysisResult

# HeadPose: TypeAlias = tuple[float, float, float]
# HeadPoseSequence: TypeAlias = list[HeadPose]


class HeadPoseActionDetector:
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
        self.head_pose_action_classifier = HeadPoseActionClassifier(
            yaw_threshold=self.yaw_threshold,
            pitch_threshold=self.pitch_threshold,
            neutral_yaw_threshold=self.neutral_yaw_threshold,
            neutral_pitch_threshold=self.neutral_pitch_threshold,
        )

    def detect(
        self,
        frames_face_analysis_results: VideoFrameAnalysisResult,
    ) -> list[HeadPoseActionType]:

        observed_actions: list[HeadPoseActionType] = []
        for face_analysis_results in frames_face_analysis_results.frames:
            frame_action = self.head_pose_action_classifier.classify(
                face_analysis_results.detection,
            )

            if frame_action is not None:
                observed_actions.append(frame_action)

        return observed_actions
