import numpy as np
import pytest
from PIL import Image

from ai_modules.face_detection.domain.analyzed_video_frame import (
    AnalyzedVideoFrame,
    VideoFrameAnalysisResult,
)
from ai_modules.face_detection.domain.face_detection import FaceDetection
from ai_modules.face_verification.application.face_verification_service import (
    FaceVerificationService,
)
from ai_modules.face_verification.domain.face_embedding import FaceEmbedding
from ai_modules.face_verification.domain.face_match_threshold_policy import (
    FaceMatchThresholdPolicy,
)
from core.exceptions import NoUsableVideoFrameError
from core.video_frame import VideoFrame
from ai_modules.face_verification.domain.cosine_face_embedding_similarity_calculator import (
    CosineFaceEmbeddingSimilarityCalculator,
)


class FakePreprocessor:
    def preprocess(
        self,
        image: Image.Image,
    ) -> Image.Image:
        return image

    def preprocess_with_detection(
        self,
        image: Image.Image,
        detection: FaceDetection,
    ) -> Image.Image:
        return image


class FakeEmbedder:
    def __init__(
        self,
        vectors: list[np.ndarray],
    ) -> None:
        self.vectors = vectors
        self.index = 0

    def extract(
        self,
        image: Image.Image,
    ) -> FaceEmbedding:
        vector = self.vectors[self.index]
        self.index += 1

        return FaceEmbedding(
            vector=vector,
            model_name="edgeface",
        )


def make_detection() -> FaceDetection:
    return FaceDetection(
        bbox=(
            0.0,
            0.0,
            100.0,
            100.0,
        ),
        landmarks=np.zeros(
            (
                5,
                2,
            ),
            dtype=np.float32,
        ),
        score=0.99,
    )


def make_video_analysis(
    frame_count: int,
) -> VideoFrameAnalysisResult:
    detection = make_detection()
    analyzed_frames = tuple(
        AnalyzedVideoFrame(
            frame=VideoFrame(
                frame_index=index,
                timestamp_seconds=float(index),
                image=Image.new(
                    "RGB",
                    (
                        112,
                        112,
                    ),
                ),
            ),
            detection=detection,
        )
        for index in range(frame_count)
    )

    return VideoFrameAnalysisResult(
        frames=analyzed_frames,
        total_frames=frame_count,
        skipped_frames=0,
    )


def make_service(
    vectors: list[np.ndarray],
) -> FaceVerificationService:
    return FaceVerificationService(
        preprocessor=FakePreprocessor(),
        embedder=FakeEmbedder(vectors=vectors),
        similarity_calculator=CosineFaceEmbeddingSimilarityCalculator(),
        threshold_policy=FaceMatchThresholdPolicy(
            level_to_threshold={
                2: 0.55,
            }
        ),
        neutral_yaw_threshold=8.0,
        neutral_pitch_threshold=10.0,
    )


def test_video_face_match_service_accepts_matching_video():
    reference_vector = np.array(
        [
            1.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    matching_vector = reference_vector.copy()

    service = make_service(
        vectors=[
            reference_vector,
            matching_vector,
            matching_vector,
            matching_vector,
        ]
    )

    result = service.verify_frames_faces_with_reference_face(
        reference_image=Image.new(
            "RGB",
            (
                112,
                112,
            ),
        ),
        reference_face_detection_result=make_detection(),
        frames_face_detection_result=make_video_analysis(
            frame_count=3,
        ),
        strictness_level=2,
    )

    assert result.is_match is True
    assert result.score == 1.0
    assert result.best_score == 1.0
    assert result.processed_frames == 3
    assert result.skipped_frames == 0
    assert result.model_name == "edgeface"


def test_video_face_match_service_rejects_non_matching_video():
    reference_vector = np.array(
        [
            1.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    different_vector = np.array(
        [
            0.0,
            1.0,
            0.0,
        ],
        dtype=np.float32,
    )

    service = make_service(
        vectors=[
            reference_vector,
            different_vector,
            different_vector,
            different_vector,
        ]
    )

    result = service.verify_frames_faces_with_reference_face(
        reference_image=Image.new(
            "RGB",
            (
                112,
                112,
            ),
        ),
        reference_face_detection_result=make_detection(),
        frames_face_detection_result=make_video_analysis(
            frame_count=3,
        ),
        strictness_level=2,
    )

    assert result.is_match is False
    assert result.score == 0.0
    assert result.best_score == 0.0


def test_video_face_match_service_rejects_video_with_no_frames():
    service = make_service(
        vectors=[
            np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
        ]
    )

    with pytest.raises(NoUsableVideoFrameError):
        service.verify_frames_faces_with_reference_face(
            reference_image=Image.new(
                "RGB",
                (
                    112,
                    112,
                ),
            ),
            reference_face_detection_result=make_detection(),
            frames_face_detection_result=make_video_analysis(
                frame_count=0,
            ),
            strictness_level=2,
        )
