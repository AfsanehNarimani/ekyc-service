from pathlib import Path

import numpy as np
from PIL import Image

from core.exceptions import (
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    NoUsableVideoFrameError,
)
from core.video_face_match_result import VideoFaceMatchResult
from face.base_embedder import BaseFaceEmbedder
from face.face_matcher import FaceMatcher
from face.face_match_threshold_policy import FaceMatchThresholdPolicy
from face.face_preprocessor import FacePreprocessor
from video.video_frame_extractor import VideoFrameExtractor


class VideoFaceMatchService:
    def __init__(
        self,
        preprocessor: FacePreprocessor,
        embedder: BaseFaceEmbedder,
        frame_extractor: VideoFrameExtractor,
        threshold_policy: FaceMatchThresholdPolicy,
    ) -> None:
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.frame_extractor = frame_extractor
        self.threshold_policy = threshold_policy

    def match_id_card_to_video(
        self,
        national_id_image: Image.Image,
        video_path: Path,
        strictness_level: int,
    ) -> VideoFaceMatchResult:
        threshold = self.threshold_policy.get_threshold(strictness_level)

        reference_face = self.preprocessor.preprocess(national_id_image)
        reference_embedding = self.embedder.extract(reference_face)

        frames = self.frame_extractor.extract(video_path)

        frame_scores: list[float] = []
        skipped_frames = 0

        matcher = FaceMatcher(
            threshold=threshold,
        )

        for frame in frames:
            try:
                candidate_face = self.preprocessor.preprocess(frame.image)
                candidate_embedding = self.embedder.extract(candidate_face)

                match_result = matcher.match(
                    reference=reference_embedding,
                    candidate=candidate_embedding,
                )

                frame_scores.append(match_result.score)

            except (NoFaceDetectedError, MultipleFacesDetectedError):
                skipped_frames += 1

        if not frame_scores:
            raise NoUsableVideoFrameError(
                "No usable video frame could be used for face matching."
            )

        median_score = float(
            np.median(
                np.array(
                    frame_scores,
                    dtype=np.float32,
                )
            )
        )
        best_score = float(max(frame_scores))

        return VideoFaceMatchResult(
            is_match=median_score >= threshold,
            score=median_score,
            best_score=best_score,
            threshold=threshold,
            model_name=reference_embedding.model_name,
            processed_frames=len(frame_scores),
            skipped_frames=skipped_frames,
            frame_scores=frame_scores,
        )
