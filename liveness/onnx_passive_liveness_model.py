from pathlib import Path

import numpy as np
import onnxruntime as ort

from core.face_detection import FaceDetection
from liveness.passive_liveness_preprocessor import (
    PassiveLivenessPreprocessor,
)


class ONNXPassiveLivenessModel:
    def __init__(
        self,
        model_path: Path,
        preprocessor: PassiveLivenessPreprocessor,
        real_class_index: int = 0,
        providers: list[str] | None = None,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Passive liveness model not found: {model_path}")

        self.preprocessor = preprocessor
        self.real_class_index = real_class_index

        self.session = ort.InferenceSession(
            str(model_path),
            providers=providers or ["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_real_score(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
    ) -> float:
        probabilities = self.predict_probabilities(
            frame_bgr=frame_bgr,
            detection=detection,
        )

        return self._extract_real_score(probabilities)

    def predict_probabilities(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
    ) -> np.ndarray:
        model_input = self.preprocessor.preprocess(
            frame_bgr=frame_bgr,
            detection=detection,
        )

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name: model_input.tensor,
            },
        )

        logits_or_probs = np.asarray(outputs[0]).reshape(-1)

        return self._to_probabilities(logits_or_probs)

    def _extract_real_score(
        self,
        probabilities: np.ndarray,
    ) -> float:
        if probabilities.size == 1:
            return float(probabilities[0])

        if self.real_class_index >= probabilities.size:
            raise ValueError(
                "real_class_index is out of range for passive liveness output."
            )

        return float(probabilities[self.real_class_index])

    def _to_probabilities(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        if self._looks_like_probabilities(values):
            return values.astype(np.float32)

        return self._softmax(values)

    def _looks_like_probabilities(
        self,
        values: np.ndarray,
    ) -> bool:
        return (
            bool(np.all(values >= 0.0))
            and bool(np.all(values <= 1.0))
            and bool(np.isclose(np.sum(values), 1.0, atol=1e-3))
        )

    def _softmax(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)

        return exp_values / np.sum(exp_values)
