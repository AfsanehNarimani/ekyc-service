from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True, slots=True)
class VideoFrame:
    frame_index: int
    timestamp_seconds: float
    image: Image.Image
