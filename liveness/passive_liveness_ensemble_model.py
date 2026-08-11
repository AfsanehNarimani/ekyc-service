from collections.abc import Sequence

import numpy as np

from ai_modules.face_detection.domain.face_detection import FaceDetection
from liveness.onnx_passive_liveness_model import ONNXPassiveLivenessModel


class PassiveLivenessEnsembleModel:
    def __init__(
        self,
        models: Sequence[ONNXPassiveLivenessModel],
        real_class_index: int = 1,
    ) -> None:
        if not models:
            raise ValueError("At least one passive liveness model is required.")

        self.models = tuple(models)
        self.real_class_index = real_class_index

    def predict_real_score(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
    ) -> float:
        probabilities = self.predict_probabilities(
            frame_bgr=frame_bgr,
            detection=detection,
        )

        if not 0 <= self.real_class_index < probabilities.size:
            raise ValueError("real_class_index is out of range for ensemble output.")

        return float(probabilities[self.real_class_index])

    def predict_probabilities(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
    ) -> np.ndarray:
        model_probabilities = [
            model.predict_probabilities(
                frame_bgr=frame_bgr,
                detection=detection,
            )
            for model in self.models
        ]

        output_shapes = {probabilities.shape for probabilities in model_probabilities}

        if len(output_shapes) != 1:
            raise ValueError("Passive liveness models have incompatible outputs.")

        return np.mean(
            np.stack(
                model_probabilities,
                axis=0,
            ),
            axis=0,
        )
