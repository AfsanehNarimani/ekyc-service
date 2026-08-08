from PIL import Image

from ai_modules.face_verification.infrastructure.preprocessing.face_aligner import FaceAligner
from ai_modules.face_verification.infrastructure.detection.insightface_detector import InsightFaceDetector
from ai_modules.face_verification.infrastructure.detection.single_face_validator import SingleFaceValidator
from core.face_detection import FaceDetection


class FacePreprocessor:
    def __init__(
        self,
        detector: InsightFaceDetector,
        validator: SingleFaceValidator,
        aligner: FaceAligner,
    ) -> None:
        self.detector = detector
        self.validator = validator
        self.aligner = aligner

    def preprocess(
        self,
        image: Image.Image,
    ) -> Image.Image:
        detections = self.detector.detect(image)
        detection = self.validator.validate(detections)

        return self.preprocess_with_detection(
            image=image,
            detection=detection,
        )

    def preprocess_with_detection(
        self,
        image: Image.Image,
        detection: FaceDetection,
    ) -> Image.Image:
        return self.aligner.align(
            image=image,
            detection=detection,
        )
