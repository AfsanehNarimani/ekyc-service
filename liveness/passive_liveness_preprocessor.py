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
        bbox_margin_ratio: float = 0.2,
    ) -> None:
        self.input_width = input_width
        self.input_height = input_height
        self.bbox_margin_ratio = bbox_margin_ratio

    def preprocess(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
    ) -> PassiveLivenessInput:
        face_crop = self._crop_face(
            frame_bgr=frame_bgr,
            detection=detection,
        )

        resized = cv2.resize(
            face_crop,
            (
                self.input_width,
                self.input_height,
            ),
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB,
        )

        normalized = rgb.astype(np.float32) / 255.0
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

    def _crop_face(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
    ) -> np.ndarray:
        image_height, image_width = frame_bgr.shape[:2]

        x1, y1, x2, y2 = detection.bbox

        box_width = x2 - x1
        box_height = y2 - y1

        margin_x = box_width * self.bbox_margin_ratio
        margin_y = box_height * self.bbox_margin_ratio

        left = max(0, int(x1 - margin_x))
        top = max(0, int(y1 - margin_y))
        right = min(image_width, int(x2 + margin_x))
        bottom = min(image_height, int(y2 + margin_y))

        if right <= left or bottom <= top:
            raise ValueError("Invalid face crop bounding box.")

        return frame_bgr[top:bottom, left:right]
