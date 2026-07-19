import cv2
import numpy as np
from PIL import Image


class EdgeFacePreprocessor:
    def preprocess(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        image_array = np.asarray(image.convert("RGB"))

        image_array = cv2.resize(
            image_array,
            (112, 112),
        )

        image_array = image_array.astype(np.float32)
        image_array = (image_array - 127.5) / 128.0

        image_array = np.transpose(
            image_array,
            (2, 0, 1),
        )

        return np.expand_dims(
            image_array,
            axis=0,
        )
