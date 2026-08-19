import math
from pathlib import Path

import numpy as np

from ai_modules.face_verification.evaluations.face_verification_metrics import (
    calculate_rate,
    select_threshold,
)
from ai_modules.face_verification.scripts.run_face_verification_evaluation import (
    load_genuine_pairs,
)


def test_select_lowest_threshold_that_satisfies_target_fmr() -> None:
    impostor_scores = np.asarray(
        [0.1, 0.2, 0.3, 0.4],
        dtype=np.float64,
    )

    threshold = select_threshold(
        impostor_scores=impostor_scores,
        target_false_match_rate=0.25,
    )

    assert threshold == 0.4
    assert (
        calculate_rate(
            scores=impostor_scores,
            threshold=threshold,
        )
        == 0.25
    )


def test_select_threshold_above_maximum_when_target_is_too_small() -> None:
    impostor_scores = np.asarray(
        [0.1, 0.2, 0.3, 0.4],
        dtype=np.float64,
    )

    threshold = select_threshold(
        impostor_scores=impostor_scores,
        target_false_match_rate=0.1,
    )

    assert threshold == math.nextafter(
        0.4,
        math.inf,
    )
    assert (
        calculate_rate(
            scores=impostor_scores,
            threshold=threshold,
        )
        == 0.0
    )


def test_select_threshold_handles_tied_scores() -> None:
    impostor_scores = np.asarray(
        [0.1, 0.2, 0.4, 0.4],
        dtype=np.float64,
    )

    threshold = select_threshold(
        impostor_scores=impostor_scores,
        target_false_match_rate=0.25,
    )

    assert threshold > 0.4
    assert (
        calculate_rate(
            scores=impostor_scores,
            threshold=threshold,
        )
        == 0.0
    )


def test_load_only_genuine_pairs_from_lfw_protocol(
    tmp_path: Path,
) -> None:
    pairs_path = tmp_path / "pairs.txt"
    images_directory = tmp_path / "lfw_funneled"

    pairs_path.write_text(
        "\n".join(
            [
                "1 1",
                "Alice 1 2",
                "Alice 1 Bob 1",
            ]
        ),
        encoding="utf-8",
    )

    pairs = load_genuine_pairs(
        pairs_path=pairs_path,
        images_directory=images_directory,
    )

    assert pairs == (
        (
            images_directory / "Alice" / "Alice_0001.jpg",
            images_directory / "Alice" / "Alice_0002.jpg",
        ),
    )
