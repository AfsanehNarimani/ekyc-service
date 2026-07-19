from abc import ABC, abstractmethod

from PIL import Image

from core.face_embedding import FaceEmbedding


class BaseFaceEmbedder(ABC):
    @abstractmethod
    def extract(
        self,
        image: Image.Image,
    ) -> FaceEmbedding:
        pass
