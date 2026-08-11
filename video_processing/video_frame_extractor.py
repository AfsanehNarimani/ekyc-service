from pathlib import Path

import cv2
from PIL import Image

from core.exceptions import InvalidVideoError
from core.video_frame import VideoFrame


class VideoFrameExtractor:
    def __init__(
        self,
        frame_interval_seconds: float = 1.0,
        max_frames: int = 30,
    ) -> None:
        if frame_interval_seconds <= 0:
            raise ValueError("frame_interval_seconds must be positive.")

        if max_frames <= 0:
            raise ValueError("max_frames must be positive.")

        self.frame_interval_seconds = frame_interval_seconds
        self.max_frames = max_frames

    def extract(
        self,
        video_path: Path,
    ) -> list[VideoFrame]:
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise InvalidVideoError("Could not open video file.")

        try:
            fps = capture.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 25.0

            frame_step = max(
                1,
                round(fps * self.frame_interval_seconds),
            )

            frames: list[VideoFrame] = []
            frame_index = 0
            #todo BENCHMARK MAX_FRAMES
            while len(frames) < self.max_frames:
                ok, frame_bgr = capture.read()

                if not ok:
                    break

                if frame_index % frame_step == 0:
                    frame_rgb = cv2.cvtColor(
                        frame_bgr,
                        cv2.COLOR_BGR2RGB,
                    )

                    timestamp_seconds = frame_index / fps

                    frames.append(
                        VideoFrame(
                            frame_index=frame_index,
                            timestamp_seconds=timestamp_seconds,
                            image=Image.fromarray(frame_rgb),
                        )
                    )

                frame_index += 1

            if not frames:
                raise InvalidVideoError("No frames could be extracted from video.")

            return frames

        finally:
            capture.release()
