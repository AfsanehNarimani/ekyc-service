from core.exceptions import MultipleFacesDetectedError, NoFaceDetectedError
from ai_modules.face_detection.domain.face_detection import FaceDetection


# TODO change it to MultiFaceHandler, add a function for handle_image and another function for handle_frames
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
