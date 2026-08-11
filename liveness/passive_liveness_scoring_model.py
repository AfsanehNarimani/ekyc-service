from typing import Protocol

import numpy as np

from ai_modules.face_detection.domain.face_detection import FaceDetection


class PassiveLivenessScoringModel(Protocol):
    def predict_real_score(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
    ) -> float: ...
