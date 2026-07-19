from pathlib import Path

import numpy as np

from core.exceptions import MultipleFacesDetectedError, NoFaceDetectedError
from core.video_liveness_result import VideoLivenessResult
from face.insightface_detector import InsightFaceDetector
from face.single_face_validator import SingleFaceValidator
from liveness.liveness_threshold_policy import LivenessThresholdPolicy
from video.video_frame_extractor import VideoFrameExtractor


class VideoLivenessService:
    def __init__(
        self,
        frame_extractor: VideoFrameExtractor,
        detector: InsightFaceDetector,
        validator: SingleFaceValidator,
        threshold_policy: LivenessThresholdPolicy,
    ) -> None:
        self.frame_extractor = frame_extractor
        self.detector = detector
        self.validator = validator
        self.threshold_policy = threshold_policy

    def check_liveness(
        self,
        video_path: Path,
        strictness_level: int,
    ) -> VideoLivenessResult:
        thresholds = self.threshold_policy.get_thresholds(strictness_level)
        frames = self.frame_extractor.extract(video_path)

        poses: list[tuple[float, float]] = []
        skipped_frames = 0

        for frame in frames:
            try:
                detections = self.detector.detect(frame.image)
                detection = self.validator.validate(detections)

                if detection.pose is None:
                    skipped_frames += 1
                    continue

                pitch, yaw, _roll = detection.pose
                poses.append(
                    (
                        float(pitch),
                        float(yaw),
                    )
                )

            except (NoFaceDetectedError, MultipleFacesDetectedError):
                skipped_frames += 1

        processed_frames = len(poses)

        if processed_frames < thresholds.min_required_frames:
            return VideoLivenessResult(
                passed=False,
                strictness_level=strictness_level,
                processed_frames=processed_frames,
                skipped_frames=skipped_frames,
                min_required_frames=thresholds.min_required_frames,
                pose_variation_score=0.0,
                min_pose_variation_score=thresholds.min_pose_variation_score,
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
                min_pose_variation_score=thresholds.min_pose_variation_score,
                reason="not_enough_pose_variation",
            )

        return VideoLivenessResult(
            passed=True,
            strictness_level=strictness_level,
            processed_frames=processed_frames,
            skipped_frames=skipped_frames,
            min_required_frames=thresholds.min_required_frames,
            pose_variation_score=pose_variation_score,
            min_pose_variation_score=thresholds.min_pose_variation_score,
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

        pitch_range = float(
            np.max(pose_array[:, 0]) - np.min(pose_array[:, 0])
        )
        yaw_range = float(
            np.max(pose_array[:, 1]) - np.min(pose_array[:, 1])
        )

        return max(
            pitch_range,
            yaw_range,
        )
