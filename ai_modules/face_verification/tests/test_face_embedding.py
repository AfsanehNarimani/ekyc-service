import math

import numpy as np
import pytest

from ai_modules.face_verification.domain.face_embedding import (
    FaceEmbedding,
)


@pytest.mark.parametrize(
    "invalid_vector_value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_reject_non_finite_vector_values(
    invalid_vector_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=(r"^Face embedding vector must contain " r"only finite values\.$"),
    ):
        FaceEmbedding(
            vector=np.array(
                [
                    1.0,
                    invalid_vector_value,
                ],
                dtype=np.float32,
            ),
            model_name="edgeface",
        )
