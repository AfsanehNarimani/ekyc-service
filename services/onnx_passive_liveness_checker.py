from pathlib import Path

import cv2
import numpy as np

from core.exceptions import MultipleFacesDetectedError, NoFaceDetectedError
from core.passive_liveness_result import PassiveLivenessResult
from face.insightface_detector import InsightFaceDetector
from face.single_face_validator import SingleFaceValidator
from liveness.base_passive_liveness_checker import BasePassiveLivenessChecker
from liveness.onnx_passive_liveness_model import ONNXPassiveLivenessModel
from liveness.passive_liveness_threshold_policy import (
    PassiveLivenessThresholdPolicy,
)
from video.video_frame_extractor import VideoFrameExtractor


class ONNXPassiveLivenessChecker(BasePassiveLivenessChecker):
    def __init__(
        self,
        frame_extractor: VideoFrameExtractor,
        detector: InsightFaceDetector,
        validator: SingleFaceValidator,
        model: ONNXPassiveLivenessModel,
        threshold_policy: PassiveLivenessThresholdPolicy,
        frame_interval_seconds: float,
        max_frames: int,
    ) -> None:
        self.frame_extractor = frame_extractor
        self.detector = detector
        self.validator = validator
        self.model = model
        self.threshold_policy = threshold_policy
        self.frame_interval_seconds = frame_interval_seconds
        self.max_frames = max_frames

    def check(
        self,
        video_path: Path,
        strictness_level: int,
    ) -> PassiveLivenessResult:
        threshold = self.threshold_policy.get_threshold(strictness_level)

        frame_scores: list[float] = []
        skipped_frames = 0

        frames = self.frame_extractor.extract(
            video_path=video_path,
        )

        for frame in frames:
            try:
                detections = self.detector.detect(frame.image)
                detection = self.validator.validate(detections)

                frame_bgr = self._pil_rgb_to_bgr(frame.image)

                score = self.model.predict_real_score(
                    frame_bgr=frame_bgr,
                    detection=detection,
                )

                frame_scores.append(score)

            except (
                NoFaceDetectedError,
                MultipleFacesDetectedError,
                ValueError,
            ):
                skipped_frames += 1

        if not frame_scores:
            return PassiveLivenessResult(
                passed=False,
                score=0.0,
                threshold=threshold,
                model_name="onnx_passive_liveness",
                processed_frames=0,
                skipped_frames=skipped_frames,
                reason="no_usable_frame_for_passive_liveness",
            )

        final_score = float(np.median(frame_scores))
        passed = final_score >= threshold

        return PassiveLivenessResult(
            passed=passed,
            score=final_score,
            threshold=threshold,
            model_name="onnx_passive_liveness",
            processed_frames=len(frame_scores),
            skipped_frames=skipped_frames,
            reason=(
                "passed_passive_liveness"
                if passed
                else "failed_passive_liveness"
            ),
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
