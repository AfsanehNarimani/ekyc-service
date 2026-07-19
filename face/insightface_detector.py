import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image

from core.face_detection import FaceDetection


class InsightFaceDetector:
    def __init__(
        self,
        det_thresh: float = 0.5,
        det_size: tuple[int, int] = (640, 640),
        providers: list[str] | None = None,
    ) -> None:
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers or ["CPUExecutionProvider"],
        )
        self.app.prepare(
            ctx_id=0,
            det_thresh=det_thresh,
            det_size=det_size,
        )

    def detect(
        self,
        image: Image.Image,
    ) -> list[FaceDetection]:
        image_rgb = np.asarray(image.convert("RGB"))
        image_bgr = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR,
        )

        faces = self.app.get(image_bgr)

        detections = []
        for face in faces:
            pose = getattr(
                face,
                "pose",
                None,
            )

            pose_tuple = None
            if pose is not None:
                pose_tuple = tuple(float(value) for value in pose)

            detections.append(
                FaceDetection(
                    bbox=tuple(float(value) for value in face.bbox),
                    landmarks=face.kps.astype(np.float32),
                    score=float(face.det_score),
                    pose=pose_tuple,
                )
            )

        return detections
