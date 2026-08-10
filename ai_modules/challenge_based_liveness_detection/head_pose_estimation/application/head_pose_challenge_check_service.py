from ai_modules.challenge_based_liveness_detection.head_pose_estimation. \
domain.head_pose_action_type import HeadPoseActionType
from ai_modules.challenge_based_liveness_detection.head_pose_estimation. \
    domain.head_pose_challenge_check_result import HeadPoseChallengeCheckResult
from ai_modules.challenge_based_liveness_detection.head_pose_estimation. \
    infrastructure.head_pose_action_classifier import HeadPoseActionClassifier
# TODO should be refactored
from core.analyzed_video_frame import VideoFrameAnalysisResult


class HeadPoseChallengeCheckService:
    def __init__(
            self,
            head_pose_action_classifier: HeadPoseActionClassifier,
    ) -> None:
        self.head_pose_action_classifier = head_pose_action_classifier

    # TODO rename the function
    def check_actions(
            self,
            video_analysis: VideoFrameAnalysisResult,
            expected_actions: list[HeadPoseActionType],
    ) -> HeadPoseChallengeCheckResult:
        observed_actions: list[HeadPoseActionType] = []

        # TODO a new class and function that gets the whole sequence of frames and pass the action sequence
        for analyzed_frame in video_analysis.frames:
            action = self.head_pose_action_classifier.classify(
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

        return HeadPoseChallengeCheckResult(
            passed=len(missing_actions) == 0,
            required_actions=expected_actions,
            observed_actions=observed_actions,
            compact_actions=compact_actions,
            missing_actions=missing_actions,
            processed_frames=len(video_analysis.frames),
            skipped_frames=video_analysis.skipped_frames,
        )

    @staticmethod
    def _remove_consecutive_duplicates(
            actions: list[HeadPoseActionType],
    ) -> list[HeadPoseActionType]:
        compact_actions: list[HeadPoseActionType] = []

        for action in actions:
            if not compact_actions or compact_actions[-1] != action:
                compact_actions.append(action)

        return compact_actions

    @staticmethod
    def _find_missing_actions_in_order(
            expected_actions: list[HeadPoseActionType],
            observed_actions: list[HeadPoseActionType],
    ) -> list[HeadPoseActionType]:
        observed_action_set = set(observed_actions)

        return [
            expected_action
            for expected_action in expected_actions
            if expected_action not in observed_action_set
        ]
