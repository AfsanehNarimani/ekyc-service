from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    vector: np.ndarray
    model_name: str

    def __post_init__(self) -> None:
        if self.vector.ndim != 1:
            raise ValueError("Face embedding vector must be one-dimensional.")

        if not np.isfinite(self.vector).all():
            raise ValueError(
                "Face embedding vector must contain only finite values.",
            )
