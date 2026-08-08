import argparse
from collections import Counter
from pathlib import Path

from config import settings
from ai_modules.face_verification.infrastructure.detection.insightface_detector import InsightFaceDetector
from ai_modules.face_verification.infrastructure.detection.single_face_validator import SingleFaceValidator
from liveness.head_pose_action_classifier import HeadPoseActionClassifier
from video_processing.video_frame_extractor import VideoFrameExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
    )

    return parser.parse_args()


def remove_consecutive_duplicates(actions: list[str]) -> list[str]:
    compact_actions: list[str] = []

    for action in actions:
        if not compact_actions or compact_actions[-1] != action:
            compact_actions.append(action)

    return compact_actions


def main() -> None:
    args = parse_args()

    frame_extractor = VideoFrameExtractor(
        frame_interval_seconds=args.frame_interval,
        max_frames=args.max_frames,
    )

    detector = InsightFaceDetector(
        det_thresh=settings.detector_threshold,
        providers=["CPUExecutionProvider"],
    )

    validator = SingleFaceValidator()

    classifier = HeadPoseActionClassifier(
        yaw_threshold=settings.head_pose_yaw_threshold,
        pitch_threshold=settings.head_pose_pitch_threshold,
        neutral_yaw_threshold=settings.head_pose_neutral_yaw_threshold,
        neutral_pitch_threshold=settings.head_pose_neutral_pitch_threshold,
    )

    frames = frame_extractor.extract(args.video)

    observed_actions: list[str] = []
    skipped = 0

    print()
    print("Video:", args.video)
    print("Frames:", len(frames))
    print()
    print("frame_index | time | pitch | yaw | roll | action")
    print("-" * 60)

    for frame in frames:
        try:
            detections = detector.detect(frame.image)
            detection = validator.validate(detections)

            if detection.pose is None:
                skipped += 1
                print(
                    f"{frame.frame_index:>11} | "
                    f"{frame.timestamp_seconds:>4.1f}s | "
                    "NO_POSE"
                )
                continue

            pitch, yaw, roll = detection.pose
            action = classifier.classify(detection)

            action_text = action.value if action is not None else "unknown"

            if action is not None:
                observed_actions.append(action_text)

            print(
                f"{frame.frame_index:>11} | "
                f"{frame.timestamp_seconds:>4.1f}s | "
                f"{pitch:>6.2f} | "
                f"{yaw:>6.2f} | "
                f"{roll:>6.2f} | "
                f"{action_text}"
            )

        except Exception as error:
            skipped += 1
            print(
                f"{frame.frame_index:>11} | "
                f"{frame.timestamp_seconds:>4.1f}s | "
                f"SKIPPED | {type(error).__name__}: {error}"
            )

    compact_actions = remove_consecutive_duplicates(observed_actions)

    print()
    print("Observed actions:")
    print(observed_actions)

    print()
    print("Compact actions:")
    print(compact_actions)

    print()
    print("Action counts:")
    print(dict(Counter(observed_actions)))

    print()
    print("Skipped frames:", skipped)


if __name__ == "__main__":
    main()
