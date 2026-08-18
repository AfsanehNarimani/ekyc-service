from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ai_modules.face_verification.domain.cosine_face_embedding_similarity_calculator import (
    CosineFaceEmbeddingSimilarityCalculator,
)
from ai_modules.face_verification.domain.face_embedding import FaceEmbedding
from ai_modules.face_verification.evaluations.face_verification_comparison_evaluator import (
    FaceVerificationComparisonEvaluator,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_sample import (
    FaceVerificationComparisonSample,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_status import (
    FaceVerificationComparisonStatus,
)
from core.exceptions import (
    MediaLoadError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)


class FakeImageFileLoader:
    def load(
        self,
        path: Path,
    ) -> Image.Image:
        return Image.new(
            mode="RGB",
            size=(112, 112),
        )


class ReferenceImageLoadingFailureImageFileLoader:
    def load(
        self,
        path: Path,
    ) -> Image.Image:
        raise MediaLoadError(f"Could not load image file: {path}")


class FakeFaceDetectionService:
    def detect_from_image(
        self,
        image: Image.Image,
    ) -> object:
        return object()

class ReferenceFaceDetectionFailureService:
    def __init__(
        self,
        error: Exception,
    ) -> None:
        self.error = error

    def detect_from_image(
        self,
        image: Image.Image,
    ) -> object:
        raise self.error

class CandidateFaceDetectionFailureService:
    def __init__(
        self,
        error: Exception,
    ) -> None:
        self.error = error
        self.detection_call_count = 0

    def detect_from_image(
        self,
        image: Image.Image,
    ) -> object:
        self.detection_call_count += 1

        if self.detection_call_count == 2:
            raise self.error

        return object()

class FakeFacePreprocessor:
    def preprocess_with_detection(
        self,
        image: Image.Image,
        detection: object,
    ) -> Image.Image:
        return image

class FaceAlignmentFailurePreprocessor:
    def __init__(
        self,
        failing_call_number: int,
    ) -> None:
        self.failing_call_number = failing_call_number
        self.preprocessing_call_count = 0

    def preprocess_with_detection(
        self,
        image: Image.Image,
        detection: object,
    ) -> Image.Image:
        self.preprocessing_call_count += 1

        if (
            self.preprocessing_call_count
            == self.failing_call_number
        ):
            raise RuntimeError(
                "Face alignment failed."
            )

        return image

class FakeFaceEmbedder:
    def __init__(
        self,
        embeddings: list[FaceEmbedding],
    ) -> None:
        self.embeddings = iter(embeddings)

    def extract(
        self,
        image: Image.Image,
    ) -> FaceEmbedding:
        return next(self.embeddings)

class FaceEmbeddingFailureEmbedder:
    def __init__(
        self,
        failing_call_number: int,
    ) -> None:
        self.failing_call_number = failing_call_number
        self.extraction_call_count = 0

    def extract(
        self,
        image: Image.Image,
    ) -> FaceEmbedding:
        self.extraction_call_count += 1

        if self.extraction_call_count == self.failing_call_number:
            raise RuntimeError(
                "Face embedding extraction failed."
            )

        return FaceEmbedding(
            vector=np.array(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            model_name="edgeface",
        )

class CandidateImageLoadingFailureImageFileLoader:
    def __init__(self) -> None:
        self.load_call_count = 0

    def load(
        self,
        path: Path,
    ) -> Image.Image:
        self.load_call_count += 1

        if self.load_call_count == 2:
            raise MediaLoadError(f"Could not load image file: {path}")

        return Image.new(
            mode="RGB",
            size=(112, 112),
        )

class FaceEmbeddingSimilarityCalculationFailureCalculator:
    def calculate_similarity(
        self,
        reference_embedding: FaceEmbedding,
        candidate_embedding: FaceEmbedding,
    ) -> float:
        raise RuntimeError(
            "Face embedding similarity calculation failed."
        )

def create_comparison_sample() -> FaceVerificationComparisonSample:
    return FaceVerificationComparisonSample(
        comparison_id="comparison-001",
        reference_subject_id="subject-001",
        candidate_subject_id="subject-001",
        reference_image_path=Path("reference.jpg"),
        candidate_image_path=Path("candidate.jpg"),
        is_genuine=True,
    )


def test_evaluate_successful_face_verification_comparison() -> None:
    reference_embedding = FaceEmbedding(
        vector=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        model_name="edgeface",
    )
    candidate_embedding = FaceEmbedding(
        vector=np.array(
            [0.8, 0.6],
            dtype=np.float32,
        ),
        model_name="edgeface",
    )

    evaluator = FaceVerificationComparisonEvaluator(
        model_name="edgeface",
        image_file_loader=FakeImageFileLoader(),
        face_detection_service=FakeFaceDetectionService(),
        face_preprocessor=FakeFacePreprocessor(),
        face_embedder=FakeFaceEmbedder(
            embeddings=[
                reference_embedding,
                candidate_embedding,
            ],
        ),
        similarity_calculator=(CosineFaceEmbeddingSimilarityCalculator()),
    )

    result = evaluator.evaluate_comparison(
        dataset_name="internal-ekyc",
        comparison_sample=create_comparison_sample(),
    )

    assert result.dataset_name == "internal-ekyc"
    assert result.model_name == "edgeface"
    assert result.status is FaceVerificationComparisonStatus.SUCCESS
    assert result.similarity_score == pytest.approx(0.8)
    assert result.processing_time_milliseconds >= 0
    assert result.failure_detail is None


def test_return_failure_when_reference_image_cannot_be_loaded() -> None:
    evaluator = FaceVerificationComparisonEvaluator(
        model_name="edgeface",
        image_file_loader=(ReferenceImageLoadingFailureImageFileLoader()),
        face_detection_service=FakeFaceDetectionService(),
        face_preprocessor=FakeFacePreprocessor(),
        face_embedder=FakeFaceEmbedder(
            embeddings=[],
        ),
        similarity_calculator=(CosineFaceEmbeddingSimilarityCalculator()),
    )

    comparison_sample = create_comparison_sample()

    result = evaluator.evaluate_comparison(
        dataset_name="internal-ekyc",
        comparison_sample=comparison_sample,
    )

    assert result.status is (
        FaceVerificationComparisonStatus.REFERENCE_IMAGE_LOAD_FAILED
    )
    assert result.similarity_score is None
    assert result.processing_time_milliseconds >= 0
    assert result.failure_detail == ("Could not load image file: reference.jpg")


def test_return_failure_when_candidate_image_cannot_be_loaded() -> None:
    evaluator = FaceVerificationComparisonEvaluator(
        model_name="edgeface",
        image_file_loader=(CandidateImageLoadingFailureImageFileLoader()),
        face_detection_service=FakeFaceDetectionService(),
        face_preprocessor=FakeFacePreprocessor(),
        face_embedder=FakeFaceEmbedder(
            embeddings=[],
        ),
        similarity_calculator=(CosineFaceEmbeddingSimilarityCalculator()),
    )

    comparison_sample = create_comparison_sample()

    result = evaluator.evaluate_comparison(
        dataset_name="internal-ekyc",
        comparison_sample=comparison_sample,
    )

    assert result.status is (
        FaceVerificationComparisonStatus.CANDIDATE_IMAGE_LOAD_FAILED
    )
    assert result.similarity_score is None
    assert result.processing_time_milliseconds >= 0
    assert result.failure_detail == ("Could not load image file: candidate.jpg")

@pytest.mark.parametrize(
    (
        "detection_error",
        "expected_status",
    ),
    [
        (
            NoFaceDetectedError(
                "No face was detected in the reference image."
            ),
            FaceVerificationComparisonStatus
            .REFERENCE_FACE_NOT_DETECTED,
        ),
        (
            MultipleFacesDetectedError(
                "Multiple faces were detected in the reference image."
            ),
            FaceVerificationComparisonStatus
            .REFERENCE_MULTIPLE_FACES_DETECTED,
        ),
    ],
)
def test_return_failure_when_reference_face_detection_fails(
    detection_error: Exception,
    expected_status: FaceVerificationComparisonStatus,
) -> None:
    evaluator = FaceVerificationComparisonEvaluator(
        model_name="edgeface",
        image_file_loader=FakeImageFileLoader(),
        face_detection_service=(
            ReferenceFaceDetectionFailureService(
                error=detection_error,
            )
        ),
        face_preprocessor=FakeFacePreprocessor(),
        face_embedder=FakeFaceEmbedder(
            embeddings=[],
        ),
        similarity_calculator=(
            CosineFaceEmbeddingSimilarityCalculator()
        ),
    )

    result = evaluator.evaluate_comparison(
        dataset_name="internal-ekyc",
        comparison_sample=create_comparison_sample(),
    )

    assert result.status is expected_status
    assert result.similarity_score is None
    assert result.processing_time_milliseconds >= 0
    assert result.failure_detail == str(detection_error)

@pytest.mark.parametrize(
    (
        "detection_error",
        "expected_status",
    ),
    [
        (
            NoFaceDetectedError(
                "No face was detected in the candidate image."
            ),
            FaceVerificationComparisonStatus
            .CANDIDATE_FACE_NOT_DETECTED,
        ),
        (
            MultipleFacesDetectedError(
                "Multiple faces were detected in the candidate image."
            ),
            FaceVerificationComparisonStatus
            .CANDIDATE_MULTIPLE_FACES_DETECTED,
        ),
    ],
)
def test_return_failure_when_candidate_face_detection_fails(
    detection_error: Exception,
    expected_status: FaceVerificationComparisonStatus,
) -> None:
    evaluator = FaceVerificationComparisonEvaluator(
        model_name="edgeface",
        image_file_loader=FakeImageFileLoader(),
        face_detection_service=(
            CandidateFaceDetectionFailureService(
                error=detection_error,
            )
        ),
        face_preprocessor=FakeFacePreprocessor(),
        face_embedder=FakeFaceEmbedder(
            embeddings=[],
        ),
        similarity_calculator=(
            CosineFaceEmbeddingSimilarityCalculator()
        ),
    )

    result = evaluator.evaluate_comparison(
        dataset_name="internal-ekyc",
        comparison_sample=create_comparison_sample(),
    )

    assert result.status is expected_status
    assert result.similarity_score is None
    assert result.processing_time_milliseconds >= 0
    assert result.failure_detail == str(detection_error)

@pytest.mark.parametrize(
    (
        "failing_call_number",
        "expected_status",
    ),
    [
        (
            1,
            FaceVerificationComparisonStatus
            .REFERENCE_FACE_ALIGNMENT_FAILED,
        ),
        (
            2,
            FaceVerificationComparisonStatus
            .CANDIDATE_FACE_ALIGNMENT_FAILED,
        ),
    ],
)
def test_return_failure_when_face_alignment_fails(
    failing_call_number: int,
    expected_status: FaceVerificationComparisonStatus,
) -> None:
    evaluator = FaceVerificationComparisonEvaluator(
        model_name="edgeface",
        image_file_loader=FakeImageFileLoader(),
        face_detection_service=FakeFaceDetectionService(),
        face_preprocessor=FaceAlignmentFailurePreprocessor(
            failing_call_number=failing_call_number,
        ),
        face_embedder=FakeFaceEmbedder(
            embeddings=[],
        ),
        similarity_calculator=(
            CosineFaceEmbeddingSimilarityCalculator()
        ),
    )

    result = evaluator.evaluate_comparison(
        dataset_name="internal-ekyc",
        comparison_sample=create_comparison_sample(),
    )

    assert result.status is expected_status
    assert result.similarity_score is None
    assert result.processing_time_milliseconds >= 0
    assert result.failure_detail == "Face alignment failed."

@pytest.mark.parametrize(
    (
        "failing_call_number",
        "expected_status",
    ),
    [
        (
            1,
            FaceVerificationComparisonStatus
            .REFERENCE_EMBEDDING_FAILED,
        ),
        (
            2,
            FaceVerificationComparisonStatus
            .CANDIDATE_EMBEDDING_FAILED,
        ),
    ],
)
def test_return_failure_when_face_embedding_extraction_fails(
    failing_call_number: int,
    expected_status: FaceVerificationComparisonStatus,
) -> None:
    evaluator = FaceVerificationComparisonEvaluator(
        model_name="edgeface",
        image_file_loader=FakeImageFileLoader(),
        face_detection_service=FakeFaceDetectionService(),
        face_preprocessor=FakeFacePreprocessor(),
        face_embedder=FaceEmbeddingFailureEmbedder(
            failing_call_number=failing_call_number,
        ),
        similarity_calculator=(
            CosineFaceEmbeddingSimilarityCalculator()
        ),
    )

    result = evaluator.evaluate_comparison(
        dataset_name="internal-ekyc",
        comparison_sample=create_comparison_sample(),
    )

    assert result.status is expected_status
    assert result.similarity_score is None
    assert result.processing_time_milliseconds >= 0
    assert result.failure_detail == (
        "Face embedding extraction failed."
    )

def test_return_unexpected_failure_when_similarity_calculation_fails(
) -> None:
    reference_embedding = FaceEmbedding(
        vector=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        model_name="edgeface",
    )
    candidate_embedding = FaceEmbedding(
        vector=np.array(
            [0.8, 0.6],
            dtype=np.float32,
        ),
        model_name="edgeface",
    )

    evaluator = FaceVerificationComparisonEvaluator(
        model_name="edgeface",
        image_file_loader=FakeImageFileLoader(),
        face_detection_service=FakeFaceDetectionService(),
        face_preprocessor=FakeFacePreprocessor(),
        face_embedder=FakeFaceEmbedder(
            embeddings=[
                reference_embedding,
                candidate_embedding,
            ],
        ),
        similarity_calculator=(
            FaceEmbeddingSimilarityCalculationFailureCalculator()
        ),
    )

    result = evaluator.evaluate_comparison(
        dataset_name="internal-ekyc",
        comparison_sample=create_comparison_sample(),
    )

    assert result.status is (
        FaceVerificationComparisonStatus.UNEXPECTED_ERROR
    )
    assert result.similarity_score is None
    assert result.processing_time_milliseconds >= 0
    assert result.failure_detail == (
        "Face embedding similarity calculation failed."
    )