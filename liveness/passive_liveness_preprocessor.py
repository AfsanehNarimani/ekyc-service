from dataclasses import dataclass

import cv2
import numpy as np

from core.face_detection import FaceDetection


@dataclass(frozen=True, slots=True)
class PassiveLivenessInput:
    tensor: np.ndarray


class PassiveLivenessPreprocessor:
    def __init__(
        self,
        input_width: int,
        input_height: int,
        crop_scale: float = 2.7,
    ) -> None:
        if crop_scale <= 1.0:
            raise ValueError("crop_scale must be greater than 1.0.")

        self.input_width = input_width
        self.input_height = input_height
        self.crop_scale = crop_scale

    def preprocess(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
    ) -> PassiveLivenessInput:
        face_crop = self.crop_face(
            frame_bgr=frame_bgr,
            detection=detection,
        )

        resized_bgr = cv2.resize(
            face_crop,
            (
                self.input_width,
                self.input_height,
            ),
        )

        normalized = resized_bgr.astype(np.float32) / 255.0

        chw = np.transpose(
            normalized,
            (
                2,
                0,
                1,
            ),
        )

        tensor = np.expand_dims(
            chw,
            axis=0,
        ).astype(np.float32)

        return PassiveLivenessInput(tensor=tensor)

    def crop_face(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
    ) -> np.ndarray:
        image_height, image_width = frame_bgr.shape[:2]

        x1, y1, x2, y2 = detection.bbox

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            raise ValueError("Invalid face bounding box.")

        center_x = x1 + box_width / 2.0
        center_y = y1 + box_height / 2.0

        crop_width = box_width * self.crop_scale
        crop_height = box_height * self.crop_scale

        left = int(round(center_x - crop_width / 2.0))
        top = int(round(center_y - crop_height / 2.0))
        right = int(round(center_x + crop_width / 2.0))
        bottom = int(round(center_y + crop_height / 2.0))

        left = max(0, left)
        top = max(0, top)
        right = min(image_width, right)
        bottom = min(image_height, bottom)

        if right <= left or bottom <= top:
            raise ValueError("Invalid passive liveness crop.")

        return frame_bgr[top:bottom, left:right]
