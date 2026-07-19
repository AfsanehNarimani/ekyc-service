from functools import lru_cache

from config import settings
from face.base_embedder import BaseFaceEmbedder
from face.face_aligner import FaceAligner
from face.face_embedder_factory import FaceEmbedderFactory
from face.face_matcher import FaceMatcher
from face.face_preprocessor import FacePreprocessor
from face.insightface_detector import InsightFaceDetector
from face.single_face_validator import SingleFaceValidator
from services.enrollment_service import EnrollmentService
from services.image_verification_service import ImageVerificationService
from storage.embedding_store import EmbeddingStore
from storage.reference_image_store import ReferenceImageStore
from liveness.challenge_service import ChallengeService
from liveness.challenge_store import ChallengeStore
from media.media_loader import MediaLoader

@lru_cache
def get_face_preprocessor() -> FacePreprocessor:
    return FacePreprocessor(
        detector=InsightFaceDetector(
            det_thresh=settings.detector_threshold,
            providers=["CPUExecutionProvider"],
        ),
        validator=SingleFaceValidator(),
        aligner=FaceAligner(),
    )


@lru_cache
def get_face_embedder() -> BaseFaceEmbedder:
    factory = FaceEmbedderFactory.from_settings(
        providers=["CPUExecutionProvider"],
    )

    return factory.create(
        model_name=settings.default_face_model,
    )


@lru_cache
def get_reference_store() -> ReferenceImageStore:
    return ReferenceImageStore(
        root_dir=settings.reference_dir,
    )


@lru_cache
def get_embedding_store() -> EmbeddingStore:
    return EmbeddingStore(
        root_dir=settings.embedding_dir,
    )


@lru_cache
def get_face_matcher() -> FaceMatcher:
    return FaceMatcher(
        threshold=settings.face_match_threshold,
    )


@lru_cache
def get_enrollment_service() -> EnrollmentService:
    return EnrollmentService(
        preprocessor=get_face_preprocessor(),
        embedder=get_face_embedder(),
        reference_store=get_reference_store(),
        embedding_store=get_embedding_store(),
    )


@lru_cache
def get_image_verification_service() -> ImageVerificationService:
    return ImageVerificationService(
        preprocessor=get_face_preprocessor(),
        embedder=get_face_embedder(),
        embedding_store=get_embedding_store(),
        matcher=get_face_matcher(),
    )

@lru_cache
def get_challenge_store() -> ChallengeStore:
    return ChallengeStore()


@lru_cache
def get_challenge_service() -> ChallengeService:
    return ChallengeService(
        store=get_challenge_store(),
        expires_in_seconds=120,
    )

@lru_cache
def get_media_loader() -> MediaLoader:
    return MediaLoader(
        temp_dir=settings.media_temp_dir,
        s3_region_name=settings.s3_region_name,
    )
