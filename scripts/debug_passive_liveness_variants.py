import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from api.dependencies import (
    get_face_detector,
    get_passive_liveness_preprocessor,
    get_single_face_validator,
    get_video_frame_extractor,
)
from config import settings


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def to_probabilities(values: np.ndarray) -> np.ndarray:
    values = values.reshape(-1)

    if (
        np.all(values >= 0.0)
        and np.all(values <= 1.0)
        and np.isclose(np.sum(values), 1.0, atol=1e-3)
    ):
        return values.astype(np.float32)

    return softmax(values)


def make_tensor(
    crop_bgr: np.ndarray,
    variant: str,
) -> np.ndarray:
    resized = cv2.resize(crop_bgr, (80, 80))

    if variant == "bgr_div255":
        image = resized.astype(np.float32) / 255.0

    elif variant == "rgb_div255":
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    elif variant == "bgr_1275":
        image = (resized.astype(np.float32) - 127.5) / 128.0

    elif variant == "rgb_1275":
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = (rgb.astype(np.float32) - 127.5) / 128.0

    else:
        raise ValueError(f"Unsupported variant: {variant}")

    chw = np.transpose(image, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-frames", type=int, default=3)
    args = parser.parse_args()

    session = ort.InferenceSession(
        str(settings.passive_liveness_model_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    frame_extractor = get_video_frame_extractor()
    detector = get_face_detector()
    validator = get_single_face_validator()
    preprocessor = get_passive_liveness_preprocessor()

    variants = [
        "bgr_div255",
        "rgb_div255",
        "bgr_1275",
        "rgb_1275",
    ]

    frames = frame_extractor.extract(Path(args.video))

    tested = 0

    for frame in frames:
        detections = detector.detect(frame.image)
        detection = validator.validate(detections)

        if detection.pose is not None:
            pitch, yaw, _roll = detection.pose
            if abs(pitch) > 30 or abs(yaw) > 30:
                continue

        frame_bgr = cv2.cvtColor(
            np.asarray(frame.image),
            cv2.COLOR_RGB2BGR,
        )

        crop_bgr = preprocessor.crop_face(
            frame_bgr=frame_bgr,
            detection=detection,
        )

        print(
            "\nframe",
            {
                "frame_index": frame.frame_index,
                "timestamp_seconds": round(frame.timestamp_seconds, 2),
                "pose": detection.pose,
            },
        )

        for variant in variants:
            tensor = make_tensor(crop_bgr=crop_bgr, variant=variant)

            output = session.run(
                [output_name],
                {
                    input_name: tensor,
                },
            )[0]

            probs = to_probabilities(np.asarray(output))

            print(
                variant,
                {
                    "class_0": round(float(probs[0]), 6),
                    "class_1": round(float(probs[1]), 6),
                    "class_2": round(float(probs[2]), 6),
                    "max_class": int(np.argmax(probs)),
                },
            )

        tested += 1

        if tested >= args.max_frames:
            break


if __name__ == "__main__":
    main()
