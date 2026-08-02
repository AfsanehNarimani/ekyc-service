from PIL import Image

from ai_modules.face_verification.domain.enrollment_result import EnrollmentResult
from ai_modules.face_verification.domain.person_id import PersonIDValidator
from face.base_embedder import BaseFaceEmbedder
from face.face_preprocessor import FacePreprocessor
from storage.embedding_store import EmbeddingStore
from storage.reference_image_store import ReferenceImageStore


class EnrollmentService:
    def __init__(
        self,
        preprocessor: FacePreprocessor,
        embedder: BaseFaceEmbedder,
        reference_store: ReferenceImageStore,
        embedding_store: EmbeddingStore,
        person_id_validator: PersonIDValidator | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.reference_store = reference_store
        self.embedding_store = embedding_store
        self.person_id_validator = person_id_validator or PersonIDValidator()

    def enroll(
        self,
        person_id: str,
        image: Image.Image,
    ) -> EnrollmentResult:
        self.person_id_validator.validate(person_id)

        aligned_face = self.preprocessor.preprocess(image)
        embedding = self.embedder.extract(aligned_face)

        reference_image_path = self.reference_store.save(
            person_id=person_id,
            image=aligned_face,
        )

        embedding_path = self.embedding_store.save(
            person_id=person_id,
            embedding=embedding,
        )

        return EnrollmentResult(
            person_id=person_id,
            model_name=embedding.model_name,
            reference_image_path=str(reference_image_path),
            embedding_path=str(embedding_path),
        )
