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
        providers: list[str] | None = None,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Passive liveness model not found: {model_path}")

        self.preprocessor = preprocessor
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

        return self._extract_real_score(logits_or_probs)

    def _extract_real_score(
        self,
        output: np.ndarray,
    ) -> float:
        if output.size == 1:
            return float(output[0])

        probabilities = self._softmax(output)

        # Convention for MiniFASNet-style exports is commonly:
        # class 0 = spoof, class 1 = real
        # If your ONNX export uses a different order, we will change this index.
        return float(probabilities[1])

    def _softmax(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)

        return exp_values / np.sum(exp_values)
