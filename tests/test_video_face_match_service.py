from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.exceptions import NoUsableVideoFrameError
from core.face_embedding import FaceEmbedding
from core.video_frame import VideoFrame
from face.face_match_threshold_policy import FaceMatchThresholdPolicy
from services.video_face_match_service import VideoFaceMatchService


class FakePreprocessor:
    def preprocess(
        self,
        image: Image.Image,
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


class FakeVideoFrameExtractor:
    def __init__(
        self,
        frames: list[VideoFrame],
    ) -> None:
        self.frames = frames

    def extract(
        self,
        video_path: Path,
    ) -> list[VideoFrame]:
        return self.frames


def make_frame(
    index: int,
) -> VideoFrame:
    return VideoFrame(
        frame_index=index,
        timestamp_seconds=float(index),
        image=Image.new("RGB", (112, 112)),
    )


def test_video_face_match_service_accepts_matching_video(tmp_path):
    reference_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    matching_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    service = VideoFaceMatchService(
        preprocessor=FakePreprocessor(),
        embedder=FakeEmbedder(
            vectors=[
                reference_vector,
                matching_vector,
                matching_vector,
                matching_vector,
            ]
        ),
        frame_extractor=FakeVideoFrameExtractor(
            frames=[
                make_frame(0),
                make_frame(1),
                make_frame(2),
            ]
        ),
        threshold_policy=FaceMatchThresholdPolicy(
            level_to_threshold={
                2: 0.55,
            }
        ),
    )

    result = service.match_id_card_to_video(
        national_id_image=Image.new("RGB", (112, 112)),
        video_path=tmp_path / "video.mp4",
        strictness_level=2,
    )

    assert result.is_match is True
    assert result.score == 1.0
    assert result.best_score == 1.0
    assert result.processed_frames == 3
    assert result.skipped_frames == 0
    assert result.model_name == "edgeface"


def test_video_face_match_service_rejects_non_matching_video(tmp_path):
    reference_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    different_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    service = VideoFaceMatchService(
        preprocessor=FakePreprocessor(),
        embedder=FakeEmbedder(
            vectors=[
                reference_vector,
                different_vector,
                different_vector,
                different_vector,
            ]
        ),
        frame_extractor=FakeVideoFrameExtractor(
            frames=[
                make_frame(0),
                make_frame(1),
                make_frame(2),
            ]
        ),
        threshold_policy=FaceMatchThresholdPolicy(
            level_to_threshold={
                2: 0.55,
            }
        ),
    )

    result = service.match_id_card_to_video(
        national_id_image=Image.new("RGB", (112, 112)),
        video_path=tmp_path / "video.mp4",
        strictness_level=2,
    )

    assert result.is_match is False
    assert result.score == 0.0
    assert result.best_score == 0.0


def test_video_face_match_service_rejects_video_with_no_frames(tmp_path):
    service = VideoFaceMatchService(
        preprocessor=FakePreprocessor(),
        embedder=FakeEmbedder(
            vectors=[
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ]
        ),
        frame_extractor=FakeVideoFrameExtractor(
            frames=[],
        ),
        threshold_policy=FaceMatchThresholdPolicy(
            level_to_threshold={
                2: 0.55,
            }
        ),
    )

    with pytest.raises(NoUsableVideoFrameError):
        service.match_id_card_to_video(
            national_id_image=Image.new("RGB", (112, 112)),
            video_path=tmp_path / "video.mp4",
            strictness_level=2,
        )
