import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from api.dependencies import (
    get_face_embedder,
    get_face_preprocessor,
    get_video_frame_analysis_service,
)
from face.face_matcher import FaceMatcher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
    )
    parser.add_argument(
        "--video",
        required=True,
    )
    args = parser.parse_args()

    preprocessor = get_face_preprocessor()
    embedder = get_face_embedder()
    analyzer = get_video_frame_analysis_service()

    reference_image = Image.open(args.image).convert("RGB")
    reference_face = preprocessor.preprocess(reference_image)
    reference_embedding = embedder.extract(reference_face)

    video_analysis = analyzer.analyze(
        video_path=Path(args.video),
    )
    matcher = FaceMatcher(threshold=0.0)

    legacy_scores: list[float] = []
    shared_scores: list[float] = []
    aligned_differences: list[float] = []

    for analyzed_frame in video_analysis.frames:
        frame = analyzed_frame.frame

        legacy_face = preprocessor.preprocess(frame.image)
        shared_face = preprocessor.preprocess_with_detection(
            image=frame.image,
            detection=analyzed_frame.detection,
        )

        legacy_array = np.asarray(
            legacy_face,
            dtype=np.float32,
        )
        shared_array = np.asarray(
            shared_face,
            dtype=np.float32,
        )

        aligned_difference = float(
            np.max(
                np.abs(
                    legacy_array - shared_array,
                )
            )
        )

        legacy_embedding = embedder.extract(legacy_face)
        shared_embedding = embedder.extract(shared_face)

        legacy_score = matcher.match(
            reference=reference_embedding,
            candidate=legacy_embedding,
        ).score
        shared_score = matcher.match(
            reference=reference_embedding,
            candidate=shared_embedding,
        ).score

        legacy_scores.append(legacy_score)
        shared_scores.append(shared_score)
        aligned_differences.append(aligned_difference)

        print(
            {
                "frame_index": frame.frame_index,
                "pose": analyzed_frame.detection.pose,
                "legacy_score": round(
                    float(legacy_score),
                    6,
                ),
                "shared_score": round(
                    float(shared_score),
                    6,
                ),
                "aligned_max_difference": round(
                    aligned_difference,
                    6,
                ),
            }
        )

    print(
        "summary",
        {
            "processed_frames": len(shared_scores),
            "legacy_median_score": round(
                float(np.median(legacy_scores)),
                6,
            ),
            "shared_median_score": round(
                float(np.median(shared_scores)),
                6,
            ),
            "max_aligned_difference": round(
                float(max(aligned_differences)),
                6,
            ),
        },
    )


if __name__ == "__main__":
    main()
