import cv2
import numpy as np
from PIL import Image

from ai_modules.face_verification.domain.face_model_profile import FaceModelProfile


class ModelPreprocessor:
    def __init__(
        self,
        profile: FaceModelProfile,
    ) -> None:
        self.profile = profile

    def preprocess(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        image_array = np.asarray(image.convert("RGB"))

        image_array = cv2.resize(
            image_array,
            self.profile.input_size,
        )

        if self.profile.color_order == "BGR":
            image_array = image_array[:, :, ::-1]

        image_array = image_array.astype(np.float32)

        if self.profile.normalization == "minus_127_5_div_128":
            image_array = (image_array - 127.5) / 128.0
        elif self.profile.normalization == "zero_to_one_minus_0_5_div_0_5":
            image_array = ((image_array / 255.0) - 0.5) / 0.5
        else:
            raise ValueError(f"Unsupported normalization: {self.profile.normalization}")

        if self.profile.input_layout == "NCHW":
            image_array = np.transpose(
                image_array,
                (2, 0, 1),
            )
        elif self.profile.input_layout != "NHWC":
            raise ValueError(f"Unsupported input layout: {self.profile.input_layout}")

        return np.expand_dims(
            image_array,
            axis=0,
        )
