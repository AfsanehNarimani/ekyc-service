import math
import pytest
from ai_modules.face_verification.evaluations.face_verification_threshold_candidate_generator import (
    FaceVerificationThresholdCandidateGenerator,
)


def test_generate_unique_sorted_threshold_candidates() -> None:
    generator = FaceVerificationThresholdCandidateGenerator()

    genuine_similarity_scores = [
        0.90,
        0.70,
        0.50,
    ]
    impostor_similarity_scores = [
        0.50,
        0.30,
        0.20,
    ]

    thresholds = generator.generate(
        genuine_similarity_scores=genuine_similarity_scores,
        impostor_similarity_scores=impostor_similarity_scores,
    )

    assert thresholds[:-1] == (
        0.20,
        0.30,
        0.50,
        0.70,
        0.90,
    )

    assert thresholds[-1] == math.nextafter(
        0.90,
        math.inf,
    )

    assert thresholds[-1] > max(
        genuine_similarity_scores
        + impostor_similarity_scores,
    )

def test_reject_empty_genuine_similarity_scores() -> None:
    generator = FaceVerificationThresholdCandidateGenerator()

    with pytest.raises(
        ValueError,
        match=r"^Genuine similarity scores must not be empty\.$",
    ):
        generator.generate(
            genuine_similarity_scores=[],
            impostor_similarity_scores=[
                0.30,
                0.20,
            ],
        )


def test_reject_empty_impostor_similarity_scores() -> None:
    generator = FaceVerificationThresholdCandidateGenerator()

    with pytest.raises(
        ValueError,
        match=r"^Impostor similarity scores must not be empty\.$",
    ):
        generator.generate(
            genuine_similarity_scores=[
                0.90,
                0.70,
            ],
            impostor_similarity_scores=[],
        )

@pytest.mark.parametrize(
    "invalid_similarity_score",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_reject_non_finite_similarity_scores(
    invalid_similarity_score: float,
) -> None:
    generator = FaceVerificationThresholdCandidateGenerator()

    with pytest.raises(
        ValueError,
        match=(
            r"^Similarity scores must contain "
            r"only finite values\.$"
        ),
    ):
        generator.generate(
            genuine_similarity_scores=[
                0.90,
                invalid_similarity_score,
            ],
            impostor_similarity_scores=[
                0.30,
                0.20,
            ],
        )