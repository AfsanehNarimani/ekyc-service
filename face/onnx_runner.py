from pathlib import Path

import numpy as np
import onnxruntime as ort


class ONNXRunner:
    def __init__(
        self,
        model_path: Path,
        providers: list[str] | None = None,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        self.session = ort.InferenceSession(
            str(model_path),
            providers=providers or ["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run(
        self,
        input_tensor: np.ndarray,
    ) -> np.ndarray:
        output = self.session.run(
            [self.output_name],
            {
                self.input_name: input_tensor,
            },
        )

        return output[0]
