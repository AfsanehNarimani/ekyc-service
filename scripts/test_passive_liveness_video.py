import argparse
from pathlib import Path

from api.dependencies import get_passive_liveness_checker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--level", type=int, default=2)

    args = parser.parse_args()

    checker = get_passive_liveness_checker()

    result = checker.check(
        video_path=Path(args.video),
        strictness_level=args.level,
    )

    print("Passive liveness result:")
    print(f"passed: {result.passed}")
    print(f"score: {result.score}")
    print(f"threshold: {result.threshold}")
    print(f"model_name: {result.model_name}")
    print(f"processed_frames: {result.processed_frames}")
    print(f"skipped_frames: {result.skipped_frames}")
    print(f"reason: {result.reason}")


if __name__ == "__main__":
    main()
