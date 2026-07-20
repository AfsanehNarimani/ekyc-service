import argparse
from pathlib import Path

import cv2
import numpy as np

from api.dependencies import (
    get_face_detector,
    get_onnx_passive_liveness_model,
    get_passive_liveness_preprocessor,
    get_single_face_validator,
    get_video_frame_extractor,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--output-dir",
        default="/tmp/passive_liveness_debug",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_extractor = get_video_frame_extractor()
    detector = get_face_detector()
    validator = get_single_face_validator()
    preprocessor = get_passive_liveness_preprocessor()
    model = get_onnx_passive_liveness_model()

    frames = frame_extractor.extract(
        video_path=Path(args.video),
    )

    for frame in frames:
        detections = detector.detect(frame.image)
        detection = validator.validate(detections)

        frame_bgr = cv2.cvtColor(
            np.asarray(frame.image),
            cv2.COLOR_RGB2BGR,
        )

        crop_bgr = preprocessor.crop_face(
            frame_bgr=frame_bgr,
            detection=detection,
        )

        crop_80_bgr = cv2.resize(
            crop_bgr,
            (
                80,
                80,
            ),
        )

        probabilities = model.predict_probabilities(
            frame_bgr=frame_bgr,
            detection=detection,
        )

        crop_path = output_dir / f"crop_{frame.frame_index}.jpg"

        cv2.imwrite(
            str(crop_path),
            crop_80_bgr,
        )

        print(
            {
                "frame_index": frame.frame_index,
                "timestamp_seconds": round(frame.timestamp_seconds, 2),
                "pose": detection.pose,
                "bbox": detection.bbox,
                "crop_path": str(crop_path),
                "class_0": round(float(probabilities[0]), 6),
                "class_1": round(float(probabilities[1]), 6),
                "class_2": round(float(probabilities[2]), 6),
                "selected_real_score": round(
                    float(probabilities[model.real_class_index]),
                    6,
                ),
            }
        )


if __name__ == "__main__":
    main()
