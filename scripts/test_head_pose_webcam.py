import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image

from config import settings
from face.insightface_detector import InsightFaceDetector
from face.single_face_validator import SingleFaceValidator
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture and save head-pose values from a webcam.",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index. The default camera index is 0.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/head_pose"),
        help="Directory used to save the generated CSV sequence.",
    )

    return parser.parse_args()


def create_output_path(
    output_dir: Path,
) -> Path:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S",
    )

    return output_dir / f"head_pose_{timestamp}.csv"


def create_detector() -> InsightFaceDetector:
    return InsightFaceDetector(
        det_thresh=settings.detector_threshold,
        model_name=settings.insightface_model_name,
        root_dir=settings.insightface_root_dir,
        providers=[
            "CPUExecutionProvider",
        ],
    )


def save_sequence_chart(
    csv_path: Path,
) -> Path:
    timestamps_seconds: list[float] = []
    pitch_values: list[float] = []
    yaw_values: list[float] = []

    with csv_path.open(
        mode="r",
        encoding="utf-8",
    ) as input_file:
        reader = csv.DictReader(input_file)

        for row in reader:
            if not row["pitch"] or not row["yaw"]:
                continue

            timestamps_seconds.append(float(row["timestamp_ms"]) / 1000.0)
            pitch_values.append(float(row["pitch"]))
            yaw_values.append(float(row["yaw"]))

    if not timestamps_seconds:
        raise RuntimeError("No valid head-pose values were recorded.")

    chart_path = csv_path.with_suffix(".png")

    figure, axis = plt.subplots(
        figsize=(12, 6),
    )

    axis.plot(
        timestamps_seconds,
        pitch_values,
        label="Pitch",
        linewidth=2,
    )

    axis.plot(
        timestamps_seconds,
        yaw_values,
        label="Yaw",
        linewidth=2,
    )

    axis.axhline(
        y=0.0,
        color="black",
        linewidth=1,
        linestyle="--",
        alpha=0.5,
    )

    axis.set_title(
        "Head Pose Sequence",
    )
    axis.set_xlabel(
        "Time (seconds)",
    )
    axis.set_ylabel(
        "Angle (degrees)",
    )
    axis.grid(
        visible=True,
        alpha=0.3,
    )
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        chart_path,
        dpi=150,
    )

    plt.close(figure)

    return chart_path


def main() -> None:
    args = parse_args()
    output_path = create_output_path(
        args.output_dir,
    )

    detector = create_detector()
    validator = SingleFaceValidator()

    camera = cv2.VideoCapture(
        args.camera_index,
    )

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera with index " f"{args.camera_index}")

    frame_index = 0
    started_at = time.monotonic()

    print()
    print("Press q to stop.")
    print("Output:", output_path)
    print()
    print("frame_index | timestamp_ms | pitch | yaw")
    print("-" * 52)

    try:
        with output_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.writer(output_file)

            writer.writerow(
                [
                    "frame_index",
                    "timestamp_ms",
                    "pitch",
                    "yaw",
                    "status",
                ]
            )

            while True:
                frame_captured, frame_bgr = camera.read()

                if not frame_captured:
                    print("Could not capture a frame.")
                    break

                timestamp_ms = (time.monotonic() - started_at) * 1000.0

                pitch: float | None = None
                yaw: float | None = None
                status = "no_pose"

                try:
                    frame_rgb = cv2.cvtColor(
                        frame_bgr,
                        cv2.COLOR_BGR2RGB,
                    )

                    image = Image.fromarray(
                        frame_rgb,
                    )

                    detections = detector.detect(
                        image,
                    )

                    detection = validator.validate(
                        detections,
                    )

                    if detection.pose is not None:
                        pitch, yaw, _roll = detection.pose
                        status = "detected"

                except Exception as error:
                    status = type(error).__name__

                writer.writerow(
                    [
                        frame_index,
                        round(timestamp_ms, 2),
                        (round(pitch, 4) if pitch is not None else ""),
                        (round(yaw, 4) if yaw is not None else ""),
                        status,
                    ]
                )

                output_file.flush()

                if pitch is not None and yaw is not None:
                    pose_text = f"Pitch: {pitch:.2f}  " f"Yaw: {yaw:.2f}"

                    print(
                        f"{frame_index:>11} | "
                        f"{timestamp_ms:>12.2f} | "
                        f"{pitch:>6.2f} | "
                        f"{yaw:>6.2f}"
                    )
                else:
                    pose_text = f"Pose unavailable: {status}"

                    print(
                        f"{frame_index:>11} | "
                        f"{timestamp_ms:>12.2f} | "
                        f"N/A | N/A | {status}"
                    )

                cv2.putText(
                    frame_bgr,
                    pose_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    "Head Pose Test Environment",
                    frame_bgr,
                )

                frame_index += 1

                pressed_key = cv2.waitKey(1) & 0xFF

                if pressed_key == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    chart_path = save_sequence_chart(
        output_path,
    )

    print()
    print("Sequence saved:")
    print(output_path)

    print()
    print("Chart saved:")
    print(chart_path)


if __name__ == "__main__":
    main()
