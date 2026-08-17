import argparse
from pathlib import Path

import cv2
import numpy as np

from api.dependencies import (
    get_face_detector,
    get_onnx_passive_liveness_model,
    get_onnx_passive_liveness_v1se_model,
    get_passive_liveness_ensemble_model,
    get_passive_liveness_preprocessor,
    get_passive_liveness_v1se_preprocessor,
    get_single_face_validator,
    get_video_frame_extractor,
)


def format_probabilities(
    probabilities: np.ndarray,
    real_class_index: int,
) -> dict[str, float]:
    return {
        "class_0": round(float(probabilities[0]), 6),
        "class_1": round(float(probabilities[1]), 6),
        "class_2": round(float(probabilities[2]), 6),
        "real_score": round(
            float(probabilities[real_class_index]),
            6,
        ),
    }


def is_neutral_pose(
    pose,
    threshold: float = 30.0,
) -> bool:
    if pose is None:
        return True

    pitch, yaw, _roll = pose

    return abs(pitch) <= threshold and abs(yaw) <= threshold


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

    v2_preprocessor = get_passive_liveness_preprocessor()
    v1se_preprocessor = get_passive_liveness_v1se_preprocessor()

    v2_model = get_onnx_passive_liveness_model()
    v1se_model = get_onnx_passive_liveness_v1se_model()
    ensemble_model = get_passive_liveness_ensemble_model()

    v2_scores: list[float] = []
    v1se_scores: list[float] = []
    ensemble_scores: list[float] = []

    frames = frame_extractor.extract(
        video_path=Path(args.video),
    )

    for frame in frames:
        detections = detector.detect(frame.image)
        detection = validator.validate(detections)

        if not is_neutral_pose(detection.pose):
            continue

        frame_bgr = cv2.cvtColor(
            np.asarray(frame.image),
            cv2.COLOR_RGB2BGR,
        )

        v2_crop = v2_preprocessor.crop_face(
            frame_bgr=frame_bgr,
            detection=detection,
        )
        v1se_crop = v1se_preprocessor.crop_face(
            frame_bgr=frame_bgr,
            detection=detection,
        )

        v2_crop_path = output_dir / f"v2_crop_{frame.frame_index}.jpg"
        v1se_crop_path = output_dir / f"v1se_crop_{frame.frame_index}.jpg"

        cv2.imwrite(
            str(v2_crop_path),
            cv2.resize(
                v2_crop,
                (
                    80,
                    80,
                ),
            ),
        )
        cv2.imwrite(
            str(v1se_crop_path),
            cv2.resize(
                v1se_crop,
                (
                    80,
                    80,
                ),
            ),
        )

        v2_probabilities = v2_model.predict_probabilities(
            frame_bgr=frame_bgr,
            detection=detection,
        )
        v1se_probabilities = v1se_model.predict_probabilities(
            frame_bgr=frame_bgr,
            detection=detection,
        )
        ensemble_probabilities = ensemble_model.predict_probabilities(
            frame_bgr=frame_bgr,
            detection=detection,
        )

        v2_score = float(v2_probabilities[v2_model.real_class_index])
        v1se_score = float(v1se_probabilities[v1se_model.real_class_index])
        ensemble_score = float(ensemble_probabilities[ensemble_model.real_class_index])

        v2_scores.append(v2_score)
        v1se_scores.append(v1se_score)
        ensemble_scores.append(ensemble_score)

        print(
            {
                "frame_index": frame.frame_index,
                "timestamp_seconds": round(
                    frame.timestamp_seconds,
                    2,
                ),
                "pose": detection.pose,
                "v2": format_probabilities(
                    v2_probabilities,
                    v2_model.real_class_index,
                ),
                "v1se": format_probabilities(
                    v1se_probabilities,
                    v1se_model.real_class_index,
                ),
                "ensemble": format_probabilities(
                    ensemble_probabilities,
                    ensemble_model.real_class_index,
                ),
                "v2_crop_path": str(v2_crop_path),
                "v1se_crop_path": str(v1se_crop_path),
            }
        )

    if not ensemble_scores:
        print("No usable neutral frames.")
        return

    print(
        "summary",
        {
            "processed_frames": len(ensemble_scores),
            "v2_median_score": round(
                float(np.median(v2_scores)),
                6,
            ),
            "v1se_median_score": round(
                float(np.median(v1se_scores)),
                6,
            ),
            "ensemble_median_score": round(
                float(np.median(ensemble_scores)),
                6,
            ),
        },
    )


if __name__ == "__main__":
    main()
