from dataclasses import dataclass

from ai_modules.face_detection.domain.face_detection import FaceDetection
from core.video_frame import VideoFrame


@dataclass(frozen=True, slots=True)
class AnalyzedVideoFrame:
    frame: VideoFrame
    detection: FaceDetection


@dataclass(frozen=True, slots=True)
class VideoFrameAnalysisResult:
    frames: tuple[AnalyzedVideoFrame, ...]
    total_frames: int
    skipped_frames: int
