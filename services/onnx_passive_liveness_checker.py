import cv2
import numpy as np

from ai_modules.face_detection.domain.analyzed_video_frame import VideoFrameAnalysisResult
from ai_modules.face_detection.domain.face_detection import FaceDetection
from core.passive_liveness_result import PassiveLivenessResult
from liveness.base_passive_liveness_checker import (
    BasePassiveLivenessChecker,
)
from liveness.passive_liveness_scoring_model import (
    PassiveLivenessScoringModel,
)
from liveness.passive_liveness_threshold_policy import (
    PassiveLivenessThresholdPolicy,
)


class ONNXPassiveLivenessChecker(BasePassiveLivenessChecker):
    def __init__(
        self,
        model: PassiveLivenessScoringModel,
        threshold_policy: PassiveLivenessThresholdPolicy,
        model_name: str = "onnx_passive_liveness",
        neutral_yaw_threshold: float = 30.0,
        neutral_pitch_threshold: float = 30.0,
    ) -> None:
        self.model = model
        self.threshold_policy = threshold_policy
        self.model_name = model_name
        self.neutral_yaw_threshold = neutral_yaw_threshold
        self.neutral_pitch_threshold = neutral_pitch_threshold

    def check(
        self,
        video_analysis: VideoFrameAnalysisResult,
        strictness_level: int,
    ) -> PassiveLivenessResult:
        threshold = self.threshold_policy.get_threshold(
            strictness_level,
        )

        frame_scores: list[float] = []
        skipped_frames = video_analysis.skipped_frames

        for analyzed_frame in video_analysis.frames:
            detection = analyzed_frame.detection

            if not self._is_neutral_pose(detection):
                skipped_frames += 1
                continue

            try:
                frame_bgr = self._pil_rgb_to_bgr(
                    analyzed_frame.frame.image,
                )

                score = self.model.predict_real_score(
                    frame_bgr=frame_bgr,
                    detection=detection,
                )

                frame_scores.append(score)

            except ValueError:
                skipped_frames += 1

        if not frame_scores:
            return PassiveLivenessResult(
                passed=False,
                score=0.0,
                threshold=threshold,
                model_name=self.model_name,
                processed_frames=0,
                skipped_frames=skipped_frames,
                reason=("no_neutral_usable_frame_for_passive_liveness"),
                frame_scores=[],
            )

        final_score = float(np.median(frame_scores))
        passed = final_score >= threshold

        return PassiveLivenessResult(
            passed=passed,
            score=final_score,
            threshold=threshold,
            model_name=self.model_name,
            processed_frames=len(frame_scores),
            skipped_frames=skipped_frames,
            reason=("passed_passive_liveness" if passed else "failed_passive_liveness"),
            frame_scores=frame_scores,
        )

    def _is_neutral_pose(
        self,
        detection: FaceDetection,
    ) -> bool:
        if detection.pose is None:
            return True

        pitch, yaw, _roll = detection.pose

        return (
            abs(yaw) <= self.neutral_yaw_threshold
            and abs(pitch) <= self.neutral_pitch_threshold
        )

    def _pil_rgb_to_bgr(
        self,
        image,
    ) -> np.ndarray:
        frame_rgb = np.asarray(image)

        return cv2.cvtColor(
            frame_rgb,
            cv2.COLOR_RGB2BGR,
        )
