from PIL import Image

from face.face_aligner import FaceAligner
from face.insightface_detector import InsightFaceDetector
from face.single_face_validator import SingleFaceValidator


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

        return self.aligner.align(
            image=image,
            detection=detection,
        )
