from core.action_check_result import ActionCheckResult
from core.analyzed_video_frame import VideoFrameAnalysisResult
from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_action import HeadPoseAction
from liveness.head_pose_action_classifier import (
    HeadPoseActionClassifier,
)


class HeadPoseChallengeCheckService:
    def __init__(
        self,
        classifier: HeadPoseActionClassifier,
    ) -> None:
        self.classifier = classifier

    def check_actions(
        self,
        video_analysis: VideoFrameAnalysisResult,
        expected_actions: list[HeadPoseAction],
    ) -> ActionCheckResult:
        observed_actions: list[HeadPoseAction] = []

        for analyzed_frame in video_analysis.frames:
            action = self.classifier.classify(
                analyzed_frame.detection,
            )

            if action is not None:
                observed_actions.append(action)

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
            processed_frames=len(video_analysis.frames),
            skipped_frames=video_analysis.skipped_frames,
        )

    def _remove_consecutive_duplicates(
        self,
        actions: list[HeadPoseAction],
    ) -> list[HeadPoseAction]:
        compact_actions: list[HeadPoseAction] = []

        for action in actions:
            if not compact_actions or compact_actions[-1] != action:
                compact_actions.append(action)

        return compact_actions

    def _find_missing_actions_in_order(
        self,
        expected_actions: list[HeadPoseAction],
        observed_actions: list[HeadPoseAction],
    ) -> list[HeadPoseAction]:
        observed_action_set = set(observed_actions)

        return [
            expected_action
            for expected_action in expected_actions
            if expected_action not in observed_action_set
        ]
