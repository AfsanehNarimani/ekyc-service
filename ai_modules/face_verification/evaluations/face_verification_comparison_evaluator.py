import time

from ai_modules.face_detection.application.face_detection_service import (
    FaceDetectionService,
)
from ai_modules.face_verification.domain.base_face_embedding_similarity_calculator import (
    BaseFaceEmbeddingSimilarityCalculator,
)
from ai_modules.face_verification.evaluations.base_face_verification_comparison_evaluator import (
    BaseFaceVerificationComparisonEvaluator,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_evaluation_result import (
    FaceVerificationComparisonEvaluationResult,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_sample import (
    FaceVerificationComparisonSample,
)
from ai_modules.face_verification.evaluations.face_verification_comparison_status import (
    FaceVerificationComparisonStatus,
)
from ai_modules.face_verification.infrastructure.embedders.base_embedder import (
    BaseFaceEmbedder,
)
from ai_modules.face_verification.infrastructure.preprocessing.face_preprocessor import (
    FacePreprocessor,
)
from media.image_file_loader import ImageFileLoader
from core.exceptions import (
    MediaLoadError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)


class FaceVerificationComparisonEvaluator(BaseFaceVerificationComparisonEvaluator):
    def __init__(
        self,
        model_name: str,
        image_file_loader: ImageFileLoader,
        face_detection_service: FaceDetectionService,
        face_preprocessor: FacePreprocessor,
        face_embedder: BaseFaceEmbedder,
        similarity_calculator: BaseFaceEmbeddingSimilarityCalculator,
    ) -> None:
        self.model_name = model_name
        self.image_file_loader = image_file_loader
        self.face_detection_service = face_detection_service
        self.face_preprocessor = face_preprocessor
        self.face_embedder = face_embedder
        self.similarity_calculator = similarity_calculator

    def evaluate_comparison(
        self,
        dataset_name: str,
        comparison_sample: FaceVerificationComparisonSample,
    ) -> FaceVerificationComparisonEvaluationResult:
        started_at = time.perf_counter()

        try:
            reference_image = self.image_file_loader.load(
                comparison_sample.reference_image_path,
            )
        except MediaLoadError as error:
            return self._create_failure_result(
                dataset_name=dataset_name,
                comparison_sample=comparison_sample,
                status=(FaceVerificationComparisonStatus.REFERENCE_IMAGE_LOAD_FAILED),
                error=error,
                started_at=started_at,
            )

        try:
            candidate_image = self.image_file_loader.load(
                comparison_sample.candidate_image_path,
            )
        except MediaLoadError as error:
            return self._create_failure_result(
                dataset_name=dataset_name,
                comparison_sample=comparison_sample,
                status=(FaceVerificationComparisonStatus.CANDIDATE_IMAGE_LOAD_FAILED),
                error=error,
                started_at=started_at,
            )

        try:
            reference_face_detection = self.face_detection_service.detect_from_image(
                reference_image,
            )
        except NoFaceDetectedError as error:
            processing_time_milliseconds = (time.perf_counter() - started_at) * 1000.0

            return FaceVerificationComparisonEvaluationResult(
                dataset_name=dataset_name,
                comparison_sample=comparison_sample,
                model_name=self.model_name,
                status=(FaceVerificationComparisonStatus.REFERENCE_FACE_NOT_DETECTED),
                processing_time_milliseconds=processing_time_milliseconds,
                failure_detail=str(error),
            )
        except MultipleFacesDetectedError as error:
            processing_time_milliseconds = (time.perf_counter() - started_at) * 1000.0

            return FaceVerificationComparisonEvaluationResult(
                dataset_name=dataset_name,
                comparison_sample=comparison_sample,
                model_name=self.model_name,
                status=(
                    FaceVerificationComparisonStatus.REFERENCE_MULTIPLE_FACES_DETECTED
                ),
                processing_time_milliseconds=processing_time_milliseconds,
                failure_detail=str(error),
            )
        try:
            candidate_face_detection = self.face_detection_service.detect_from_image(
                candidate_image,
            )
        except NoFaceDetectedError as error:
            return self._create_failure_result(
                dataset_name=dataset_name,
                comparison_sample=comparison_sample,
                status=(FaceVerificationComparisonStatus.CANDIDATE_FACE_NOT_DETECTED),
                error=error,
                started_at=started_at,
            )
        except MultipleFacesDetectedError as error:
            return self._create_failure_result(
                dataset_name=dataset_name,
                comparison_sample=comparison_sample,
                status=(
                    FaceVerificationComparisonStatus.CANDIDATE_MULTIPLE_FACES_DETECTED
                ),
                error=error,
                started_at=started_at,
            )

        try:
            aligned_reference_face = self.face_preprocessor.preprocess_with_detection(
                image=reference_image,
                detection=reference_face_detection,
            )
        except Exception as error:
            return self._create_failure_result(
                dataset_name=dataset_name,
                comparison_sample=comparison_sample,
                status=(
                    FaceVerificationComparisonStatus.REFERENCE_FACE_ALIGNMENT_FAILED
                ),
                error=error,
                started_at=started_at,
            )
        try:
            aligned_candidate_face = self.face_preprocessor.preprocess_with_detection(
                image=candidate_image,
                detection=candidate_face_detection,
            )
        except Exception as error:
            return self._create_failure_result(
                dataset_name=dataset_name,
                comparison_sample=comparison_sample,
                status=(
                    FaceVerificationComparisonStatus.CANDIDATE_FACE_ALIGNMENT_FAILED
                ),
                error=error,
                started_at=started_at,
            )
        try:
            reference_embedding = self.face_embedder.extract(
                aligned_reference_face,
            )
        except Exception as error:
            return self._create_failure_result(
                dataset_name=dataset_name,
                comparison_sample=comparison_sample,
                status=(FaceVerificationComparisonStatus.REFERENCE_EMBEDDING_FAILED),
                error=error,
                started_at=started_at,
            )

        try:
            candidate_embedding = self.face_embedder.extract(
                aligned_candidate_face,
            )

        except Exception as error:
            return self._create_failure_result(
                dataset_name=dataset_name,
                comparison_sample=comparison_sample,
                status=(FaceVerificationComparisonStatus.CANDIDATE_EMBEDDING_FAILED),
                error=error,
                started_at=started_at,
            )

        try:
            similarity_score = self.similarity_calculator.calculate_similarity(
                reference_embedding=reference_embedding,
                candidate_embedding=candidate_embedding,
            )

        except Exception as error:
            return self._create_failure_result(
                dataset_name=dataset_name,
                comparison_sample=comparison_sample,
                status=FaceVerificationComparisonStatus.UNEXPECTED_ERROR,
                error=error,
                started_at=started_at,
            )

        processing_time_milliseconds = (time.perf_counter() - started_at) * 1000.0

        return FaceVerificationComparisonEvaluationResult(
            dataset_name=dataset_name,
            comparison_sample=comparison_sample,
            model_name=self.model_name,
            status=FaceVerificationComparisonStatus.SUCCESS,
            processing_time_milliseconds=processing_time_milliseconds,
            similarity_score=similarity_score,
        )

    def _create_failure_result(
        self,
        dataset_name: str,
        comparison_sample: FaceVerificationComparisonSample,
        status: FaceVerificationComparisonStatus,
        error: Exception,
        started_at: float,
    ) -> FaceVerificationComparisonEvaluationResult:
        processing_time_milliseconds = (time.perf_counter() - started_at) * 1000.0

        return FaceVerificationComparisonEvaluationResult(
            dataset_name=dataset_name,
            comparison_sample=comparison_sample,
            model_name=self.model_name,
            status=status,
            processing_time_milliseconds=processing_time_milliseconds,
            failure_detail=str(error),
        )
