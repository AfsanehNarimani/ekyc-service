import numpy as np

from ai_modules.face_detection.domain.analyzed_video_frame import VideoFrameAnalysisResult
from core.video_liveness_result import VideoLivenessResult
from liveness.liveness_threshold_policy import (
    LivenessThresholdPolicy,
)


class VideoLivenessService:
    def __init__(
        self,
        threshold_policy: LivenessThresholdPolicy,
    ) -> None:
        self.threshold_policy = threshold_policy

    def check_liveness(
        self,
        video_analysis: VideoFrameAnalysisResult,
        strictness_level: int,
    ) -> VideoLivenessResult:
        thresholds = self.threshold_policy.get_thresholds(
            strictness_level,
        )

        poses: list[tuple[float, float]] = []
        skipped_frames = video_analysis.skipped_frames

        for analyzed_frame in video_analysis.frames:
            pose = analyzed_frame.detection.pose

            if pose is None:
                skipped_frames += 1
                continue

            pitch, yaw, _roll = pose

            poses.append(
                (
                    float(pitch),
                    float(yaw),
                )
            )

        processed_frames = len(poses)

        if processed_frames < thresholds.min_required_frames:
            return VideoLivenessResult(
                passed=False,
                strictness_level=strictness_level,
                processed_frames=processed_frames,
                skipped_frames=skipped_frames,
                min_required_frames=thresholds.min_required_frames,
                pose_variation_score=0.0,
                min_pose_variation_score=(thresholds.min_pose_variation_score),
                reason="not_enough_usable_frames",
            )

        pose_variation_score = self._calculate_pose_variation_score(poses)

        if pose_variation_score < thresholds.min_pose_variation_score:
            return VideoLivenessResult(
                passed=False,
                strictness_level=strictness_level,
                processed_frames=processed_frames,
                skipped_frames=skipped_frames,
                min_required_frames=thresholds.min_required_frames,
                pose_variation_score=pose_variation_score,
                min_pose_variation_score=(thresholds.min_pose_variation_score),
                reason="not_enough_pose_variation",
            )

        return VideoLivenessResult(
            passed=True,
            strictness_level=strictness_level,
            processed_frames=processed_frames,
            skipped_frames=skipped_frames,
            min_required_frames=thresholds.min_required_frames,
            pose_variation_score=pose_variation_score,
            min_pose_variation_score=(thresholds.min_pose_variation_score),
            reason="passed_basic_liveness",
        )

    def _calculate_pose_variation_score(
        self,
        poses: list[tuple[float, float]],
    ) -> float:
        pose_array = np.array(
            poses,
            dtype=np.float32,
        )

        pitch_range = float(np.max(pose_array[:, 0]) - np.min(pose_array[:, 0]))
        yaw_range = float(np.max(pose_array[:, 1]) - np.min(pose_array[:, 1]))

        return max(
            pitch_range,
            yaw_range,
        )
