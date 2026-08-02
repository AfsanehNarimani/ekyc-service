from PIL import Image

from core.analyzed_video_frame import VideoFrameAnalysisResult
from core.exceptions import NoUsableVideoFrameError
from core.face_detection import FaceDetection
from ai_modules.face_verification.domain.video_face_match_result import VideoFaceMatchResult
from face.base_embedder import BaseFaceEmbedder
from ai_modules.face_verification.domain.face_match_threshold_policy import (
    FaceMatchThresholdPolicy,
)
from ai_modules.face_verification.domain.face_matcher import FaceMatcher
from face.face_preprocessor import FacePreprocessor


class VideoFaceMatchService:
    def __init__(
        self,
        preprocessor: FacePreprocessor,
        embedder: BaseFaceEmbedder,
        threshold_policy: FaceMatchThresholdPolicy,
        neutral_yaw_threshold: float,
        neutral_pitch_threshold: float,
    ) -> None:
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.threshold_policy = threshold_policy
        self.neutral_yaw_threshold = neutral_yaw_threshold
        self.neutral_pitch_threshold = neutral_pitch_threshold

    def match_id_card_to_video(
        self,
        national_id_image: Image.Image,
        video_analysis: VideoFrameAnalysisResult,
        strictness_level: int,
    ) -> VideoFaceMatchResult:
        threshold = self.threshold_policy.get_threshold(
            strictness_level,
        )

        reference_face = self.preprocessor.preprocess(
            national_id_image,
        )
        reference_embedding = self.embedder.extract(
            reference_face,
        )

        frame_scores: list[float] = []
        skipped_frames = video_analysis.skipped_frames

        matcher = FaceMatcher(
            threshold=threshold,
        )

        for analyzed_frame in video_analysis.frames:
            try:
                if not self._is_neutral_pose(
                    analyzed_frame.detection,
                ):
                    skipped_frames += 1
                    continue

                candidate_face = self.preprocessor.preprocess_with_detection(
                    image=analyzed_frame.frame.image,
                    detection=analyzed_frame.detection,
                )
                candidate_embedding = self.embedder.extract(
                    candidate_face,
                )

                match_result = matcher.match(
                    reference=reference_embedding,
                    candidate=candidate_embedding,
                )

                frame_scores.append(match_result.score)

            except ValueError:
                skipped_frames += 1

        if not frame_scores:
            raise NoUsableVideoFrameError(
                "No usable video frame could be used for face matching."
            )

        best_score = float(max(frame_scores))

        return VideoFaceMatchResult(
            is_match=best_score >= threshold,
            score=best_score,
            best_score=best_score,
            threshold=threshold,
            model_name=reference_embedding.model_name,
            processed_frames=len(frame_scores),
            skipped_frames=skipped_frames,
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
