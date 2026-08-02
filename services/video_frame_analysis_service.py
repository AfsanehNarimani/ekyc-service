from pathlib import Path

from core.analyzed_video_frame import (
    AnalyzedVideoFrame,
    VideoFrameAnalysisResult,
)
from core.exceptions import (
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from face.insightface_detector import InsightFaceDetector
from face.single_face_validator import SingleFaceValidator
from video_processing.video_frame_extractor import VideoFrameExtractor


class VideoFrameAnalysisService:
    def __init__(
        self,
        frame_extractor: VideoFrameExtractor,
        detector: InsightFaceDetector,
        validator: SingleFaceValidator,
    ) -> None:
        self.frame_extractor = frame_extractor
        self.detector = detector
        self.validator = validator

    def analyze(
        self,
        video_path: Path,
    ) -> VideoFrameAnalysisResult:
        frames = self.frame_extractor.extract(video_path)

        analyzed_frames: list[AnalyzedVideoFrame] = []
        skipped_frames = 0

        for frame in frames:
            try:
                detections = self.detector.detect(frame.image)
                detection = self.validator.validate(detections)

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
