from PIL import Image

from ai_modules.face_detection.domain.analyzed_video_frame import (
    AnalyzedVideoFrame,
    VideoFrameAnalysisResult,
)
from ai_modules.face_detection.domain.face_detection import FaceDetection
from ai_modules.face_detection.domain.single_face_validator import (
    SingleFaceValidator,
)
from ai_modules.face_detection.infrastructure.insightface_detector import (
    InsightFaceDetector,
)
# TODO Should be moved to the root/domain or ai_modules/face_detection/domain
from core.exceptions import (
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from core.video_frame import VideoFrame


class FaceDetectionService:
    def __init__(
        self,
        detector: InsightFaceDetector,
        validator: SingleFaceValidator,
    ) -> None:
        self.detector = detector
        self.validator = validator

    def detect_from_image(
        self,
        image: Image.Image,
    ) -> FaceDetection:
        detections = self.detector.detect(image)

        return self.validator.validate(detections)

    def detect_from_frames(
        self,
        frames: list[VideoFrame],
    ) -> VideoFrameAnalysisResult:
        analyzed_frames: list[AnalyzedVideoFrame] = []
        skipped_frames = 0

        for frame in frames:
            try:
                # TODO do not use detect_from_image to be able to use MultiFaceHandler.handle_frames function
                detection = self.detect_from_image(
                    image=frame.image,
                )

                analyzed_frames.append(
                    AnalyzedVideoFrame(
                        frame=frame,
                        detection=detection,
                    )
                )

            except (
                NoFaceDetectedError,
                MultipleFacesDetectedError,
            ):
                skipped_frames += 1

        return VideoFrameAnalysisResult(
            frames=tuple(analyzed_frames),
            total_frames=len(frames),
            skipped_frames=skipped_frames,
        )
