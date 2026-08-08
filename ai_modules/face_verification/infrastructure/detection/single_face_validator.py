from core.exceptions import MultipleFacesDetectedError, NoFaceDetectedError
from core.face_detection import FaceDetection


class SingleFaceValidator:
    def validate(
        self,
        detections: list[FaceDetection],
    ) -> FaceDetection:
        if len(detections) == 0:
            raise NoFaceDetectedError("No face detected.")

        if len(detections) > 1:
            raise MultipleFacesDetectedError("Multiple faces detected.")

        return detections[0]
