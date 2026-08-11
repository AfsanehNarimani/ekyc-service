from abc import ABC, abstractmethod

from PIL import Image

from ai_modules.face_verification.domain.face_embedding import FaceEmbedding


class BaseFaceEmbedder(ABC):
    @abstractmethod
    def extract(
        self,
        image: Image.Image,
    ) -> FaceEmbedding:
        pass
