from pathlib import Path

from core.action_check_result import ActionCheckResult
from core.exceptions import MultipleFacesDetectedError, NoFaceDetectedError
from core.liveness_action import LivenessAction
from face.insightface_detector import InsightFaceDetector
from face.single_face_validator import SingleFaceValidator
from liveness.head_pose_action_classifier import HeadPoseActionClassifier
from video.video_frame_extractor import VideoFrameExtractor


class VideoActionCheckService:
    def __init__(
        self,
        frame_extractor: VideoFrameExtractor,
        detector: InsightFaceDetector,
        validator: SingleFaceValidator,
        classifier: HeadPoseActionClassifier,
    ) -> None:
        self.frame_extractor = frame_extractor
        self.detector = detector
        self.validator = validator
        self.classifier = classifier

    def check_actions(
        self,
        video_path: Path,
        expected_actions: list[LivenessAction],
    ) -> ActionCheckResult:
        frames = self.frame_extractor.extract(video_path)

        observed_actions: list[LivenessAction] = []
        skipped_frames = 0

        for frame in frames:
            try:
                detections = self.detector.detect(frame.image)
                detection = self.validator.validate(detections)
                action = self.classifier.classify(detection)

                if action is not None:
                    observed_actions.append(action)

            except (NoFaceDetectedError, MultipleFacesDetectedError):
                skipped_frames += 1

        compact_actions = self._remove_consecutive_duplicates(
            observed_actions,
        )

        missing_actions = self._find_missing_actions_in_order(
            expected_actions=expected_actions,
            observed_actions=compact_actions,
        )

        return ActionCheckResult(
            passed=len(missing_actions) == 0,
            required_actions=expected_actions,
            observed_actions=observed_actions,
            compact_actions=compact_actions,
            missing_actions=missing_actions,
            processed_frames=len(frames) - skipped_frames,
            skipped_frames=skipped_frames,
        )

    def _remove_consecutive_duplicates(
        self,
        actions: list[LivenessAction],
    ) -> list[LivenessAction]:
        compact_actions: list[LivenessAction] = []

        for action in actions:
            if not compact_actions or compact_actions[-1] != action:
                compact_actions.append(action)

        return compact_actions

    def _find_missing_actions_in_order(
            self,
            expected_actions: list[LivenessAction],
            observed_actions: list[LivenessAction],
    ) -> list[LivenessAction]:
        observed_action_set = set(observed_actions)

        return [
            expected_action
            for expected_action in expected_actions
            if expected_action not in observed_action_set
        ]
