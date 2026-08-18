from PIL import Image

from ai_modules.face_detection.domain.analyzed_video_frame import (
    VideoFrameAnalysisResult,
)
from ai_modules.face_detection.domain.face_detection import FaceDetection
from ai_modules.face_verification.domain.face_match_threshold_policy import (
    FaceMatchThresholdPolicy,
)
from ai_modules.face_verification.domain.face_matcher import FaceMatcher
from ai_modules.face_verification.domain.video_face_match_result import (
    VideoFaceMatchResult,
)
from ai_modules.face_verification.infrastructure.embedders.base_embedder import (
    BaseFaceEmbedder,
)
from ai_modules.face_verification.infrastructure.preprocessing.face_preprocessor import (
    FacePreprocessor,
)
from core.exceptions import NoUsableVideoFrameError
from ai_modules.face_verification.domain.base_face_embedding_similarity_calculator import (
    BaseFaceEmbeddingSimilarityCalculator,
)


class FaceVerificationService:
    def __init__(
        self,
        preprocessor: FacePreprocessor,
        embedder: BaseFaceEmbedder,
        similarity_calculator: BaseFaceEmbeddingSimilarityCalculator,
        threshold_policy: FaceMatchThresholdPolicy,
        neutral_yaw_threshold: float,
        neutral_pitch_threshold: float,
    ) -> None:
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.similarity_calculator = similarity_calculator
        self.threshold_policy = threshold_policy
        self.neutral_yaw_threshold = neutral_yaw_threshold
        self.neutral_pitch_threshold = neutral_pitch_threshold

    def verify_frames_faces_with_reference_face(
        self,
        reference_image: Image.Image,
        reference_face_detection_result: FaceDetection,
        frames_face_detection_result: VideoFrameAnalysisResult,
        strictness_level: int,
    ) -> VideoFaceMatchResult:
        threshold = self.threshold_policy.get_threshold(
            strictness_level,
        )

        reference_face = self.preprocessor.preprocess_with_detection(
            image=reference_image,
            detection=reference_face_detection_result,
        )
        reference_embedding = self.embedder.extract(
            reference_face,
        )

        frame_scores: list[float] = []
        skipped_frames = frames_face_detection_result.skipped_frames

        matcher = FaceMatcher(
            threshold=threshold,
            similarity_calculator=self.similarity_calculator,
        )

        for analyzed_frame in frames_face_detection_result.frames:
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
