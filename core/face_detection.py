from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FaceDetection:
    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray
    score: float
    pose: tuple[float, float, float] | None = None
