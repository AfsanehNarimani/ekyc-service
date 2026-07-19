from PIL import Image

from core.image_verification_result import ImageVerificationResult
from core.person_id import PersonIDValidator
from face.base_embedder import BaseFaceEmbedder
from face.face_matcher import FaceMatcher
from face.face_preprocessor import FacePreprocessor
from storage.embedding_store import EmbeddingStore


class ImageVerificationService:
    def __init__(
        self,
        preprocessor: FacePreprocessor,
        embedder: BaseFaceEmbedder,
        embedding_store: EmbeddingStore,
        matcher: FaceMatcher,
        person_id_validator: PersonIDValidator | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.embedding_store = embedding_store
        self.matcher = matcher
        self.person_id_validator = person_id_validator or PersonIDValidator()

    def verify(
        self,
        person_id: str,
        image: Image.Image,
    ) -> ImageVerificationResult:
        self.person_id_validator.validate(person_id)

        if not self.embedding_store.exists(person_id):
            raise FileNotFoundError(
                f"No enrolled embedding found for person_id={person_id}"
            )

        aligned_face = self.preprocessor.preprocess(image)
        candidate_embedding = self.embedder.extract(aligned_face)

        reference_embedding = self.embedding_store.load(
            person_id=person_id,
            model_name=candidate_embedding.model_name,
        )

        match_result = self.matcher.match(
            reference=reference_embedding,
            candidate=candidate_embedding,
        )

        return ImageVerificationResult(
            person_id=person_id,
            is_match=match_result.is_match,
            score=match_result.score,
            threshold=match_result.threshold,
            model_name=candidate_embedding.model_name,
        )
