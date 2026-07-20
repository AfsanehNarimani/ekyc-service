import onnxruntime as ort

from config import settings


def main() -> None:
    model_path = settings.passive_liveness_model_path

    print(f"Model path: {model_path}")
    print(f"Exists: {model_path.exists()}")

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    print("\nInputs:")
    for model_input in session.get_inputs():
        print(
            {
                "name": model_input.name,
                "shape": model_input.shape,
                "type": model_input.type,
            }
        )

    print("\nOutputs:")
    for model_output in session.get_outputs():
        print(
            {
                "name": model_output.name,
                "shape": model_output.shape,
                "type": model_output.type,
            }
        )


if __name__ == "__main__":
    main()
