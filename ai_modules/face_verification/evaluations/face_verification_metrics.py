import math

import numpy as np


LEVEL_TARGETS = {
    1: 1e-3,
    2: 1e-4,
    3: 1e-5,
}


def calculate_rate(
    scores: np.ndarray,
    threshold: float,
) -> float:
    return float(
        np.mean(scores >= threshold)
    )


def select_threshold(
    impostor_scores: np.ndarray,
    target_false_match_rate: float,
) -> float:
    unique_scores, score_counts = np.unique(
        impostor_scores,
        return_counts=True,
    )

    accepted_counts = np.cumsum(
        score_counts[::-1]
    )[::-1]

    false_match_rates = (
        accepted_counts
        / impostor_scores.size
    )

    eligible_indices = np.flatnonzero(
        false_match_rates
        <= target_false_match_rate
    )

    if eligible_indices.size == 0:
        return math.nextafter(
            float(unique_scores[-1]),
            math.inf,
        )

    return float(
        unique_scores[
            eligible_indices[0]
        ]
    )