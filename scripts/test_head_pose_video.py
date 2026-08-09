import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from PIL import Image

from config import settings
from ai_modules.face_verification.infrastructure.detection.insightface_detector import InsightFaceDetector
from ai_modules.face_verification.infrastructure.detection.single_face_validator import SingleFaceValidator
from ai_modules.challenge_based_liveness_detection.head_pose_estimation.infrastructure.head_pose_action_classifier import (
    HeadPoseActionClassifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract head-pose values from a video and "
            "generate CSV, chart, and annotated video."
        ),
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the input video.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/head_pose"),
        help="Directory used to save generated files.",
    )

    return parser.parse_args()


def create_output_paths(
    input_video_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S",
    )

    output_name = f"{input_video_path.stem}" f"_head_pose_{timestamp}"

    csv_path = output_dir / f"{output_name}.csv"
    chart_path = output_dir / f"{output_name}.png"
    video_path = output_dir / f"{output_name}.mp4"

    return (
        csv_path,
        chart_path,
        video_path,
    )


def create_detector() -> InsightFaceDetector:
    return InsightFaceDetector(
        det_thresh=settings.detector_threshold,
        model_name=settings.insightface_model_name,
        root_dir=settings.insightface_root_dir,
        providers=[
            "CPUExecutionProvider",
        ],
    )


def create_classifier() -> HeadPoseActionClassifier:
    return HeadPoseActionClassifier(
        yaw_threshold=settings.head_pose_yaw_threshold,
        pitch_threshold=settings.head_pose_pitch_threshold,
        neutral_yaw_threshold=(settings.head_pose_neutral_yaw_threshold),
        neutral_pitch_threshold=(settings.head_pose_neutral_pitch_threshold),
    )


def save_chart(
    timestamps_seconds: list[float],
    pitch_values: list[float],
    yaw_values: list[float],
    chart_path: Path,
) -> None:
    if not timestamps_seconds:
        raise RuntimeError("No valid head-pose values were extracted.")

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
        linestyle="--",
        linewidth=1,
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


def draw_text(
    frame,
    pose_text: str,
    frame_text: str,
) -> None:
    cv2.rectangle(
        frame,
        (10, 10),
        (620, 90),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        frame,
        pose_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        frame_text,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    input_video_path = args.video.expanduser().resolve()

    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: " f"{input_video_path}")

    if not input_video_path.is_file():
        raise ValueError(f"Input video path is not a file: " f"{input_video_path}")

    csv_path, chart_path, annotated_video_path = create_output_paths(
        input_video_path=input_video_path,
        output_dir=args.output_dir,
    )

    detector = create_detector()
    validator = SingleFaceValidator()
    classifier = create_classifier()

    video_capture = cv2.VideoCapture(
        str(input_video_path),
    )

    if not video_capture.isOpened():
        raise RuntimeError(f"Could not open input video: " f"{input_video_path}")

    fps = video_capture.get(
        cv2.CAP_PROP_FPS,
    )

    frame_width = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_WIDTH,
        )
    )

    frame_height = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT,
        )
    )

    if fps <= 0:
        fps = 30.0

    if frame_width <= 0 or frame_height <= 0:
        video_capture.release()

        raise RuntimeError("Could not determine input video dimensions.")

    codec = cv2.VideoWriter_fourcc(
        *"mp4v",
    )

    video_writer = cv2.VideoWriter(
        str(annotated_video_path),
        codec,
        fps,
        (
            frame_width,
            frame_height,
        ),
    )

    if not video_writer.isOpened():
        video_capture.release()

        raise RuntimeError(
            f"Could not create annotated video: " f"{annotated_video_path}"
        )

    timestamps_seconds: list[float] = []
    pitch_values: list[float] = []
    yaw_values: list[float] = []

    frame_index = 0

    print()
    print("Input video:", input_video_path)
    print("Input FPS:", round(fps, 2))
    print()
    print("frame_index | time | pitch | yaw | action | status")
    print("-" * 72)

    try:
        with csv_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(
                csv_file,
            )

            writer.writerow(
                [
                    "frame_index",
                    "timestamp_ms",
                    "pitch",
                    "yaw",
                    "roll",
                    "action",
                    "status",
                ]
            )

            while True:
                frame_available, frame_bgr = video_capture.read()

                if not frame_available:
                    break

                timestamp_seconds = frame_index / fps
                timestamp_ms = timestamp_seconds * 1000.0

                pitch: float | None = None
                yaw: float | None = None
                roll: float | None = None
                action_text = "unknown"
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

                    x_min, y_min, x_max, y_max = (
                        int(value) for value in detection.bbox
                    )

                    cv2.rectangle(
                        frame_bgr,
                        (
                            x_min,
                            y_min,
                        ),
                        (
                            x_max,
                            y_max,
                        ),
                        (
                            255,
                            180,
                            0,
                        ),
                        2,
                    )

                    if detection.pose is not None:
                        pitch, yaw, roll = detection.pose

                        action = classifier.classify(
                            detection,
                        )

                        if action is not None:
                            action_text = action.value

                        status = "detected"

                        timestamps_seconds.append(
                            timestamp_seconds,
                        )
                        pitch_values.append(
                            pitch,
                        )
                        yaw_values.append(
                            yaw,
                        )

                except Exception as error:
                    status = type(error).__name__

                if pitch is not None and yaw is not None:
                    pose_text = (
                        f"Pitch: {pitch:.2f}  "
                        f"Yaw: {yaw:.2f}  "
                        f"Action: {action_text}"
                    )
                else:
                    pose_text = f"Pose unavailable: {status}"

                frame_text = f"Frame: {frame_index}  " f"Time: {timestamp_seconds:.2f}s"

                draw_text(
                    frame=frame_bgr,
                    pose_text=pose_text,
                    frame_text=frame_text,
                )

                video_writer.write(
                    frame_bgr,
                )

                writer.writerow(
                    [
                        frame_index,
                        round(timestamp_ms, 2),
                        (round(pitch, 4) if pitch is not None else ""),
                        (round(yaw, 4) if yaw is not None else ""),
                        (round(roll, 4) if roll is not None else ""),
                        action_text,
                        status,
                    ]
                )

                if pitch is not None and yaw is not None:
                    print(
                        f"{frame_index:>11} | "
                        f"{timestamp_seconds:>5.2f}s | "
                        f"{pitch:>7.2f} | "
                        f"{yaw:>7.2f} | "
                        f"{action_text:>7} | "
                        f"{status}"
                    )
                else:
                    print(
                        f"{frame_index:>11} | "
                        f"{timestamp_seconds:>5.2f}s | "
                        f"{'N/A':>7} | "
                        f"{'N/A':>7} | "
                        f"{action_text:>7} | "
                        f"{status}"
                    )

                frame_index += 1

    finally:
        video_capture.release()
        video_writer.release()

    save_chart(
        timestamps_seconds=timestamps_seconds,
        pitch_values=pitch_values,
        yaw_values=yaw_values,
        chart_path=chart_path,
    )

    print()
    print("Processing completed.")
    print()
    print("CSV:")
    print(csv_path)
    print()
    print("Chart:")
    print(chart_path)
    print()
    print("Annotated video:")
    print(annotated_video_path)


if __name__ == "__main__":
    main()
