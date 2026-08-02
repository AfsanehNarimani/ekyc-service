import numpy as np
from insightface.utils import face_align
from PIL import Image

from core.face_detection import FaceDetection


class FaceAligner:
    def __init__(
        self,
        image_size: int = 112,
    ) -> None:
        self.image_size = image_size

    def align(
        self,
        image: Image.Image,
        detection: FaceDetection,
    ) -> Image.Image:
        image_rgb = np.asarray(image.convert("RGB"))

        aligned = face_align.norm_crop(
            img=image_rgb,
            landmark=detection.landmarks,
            image_size=self.image_size,
        )

        return Image.fromarray(aligned)
