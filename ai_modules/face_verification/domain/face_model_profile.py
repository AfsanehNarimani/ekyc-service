from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FaceModelProfile:
    name: str
    input_size: tuple[int, int]
    input_layout: str
    color_order: str
    normalization: str
    embedding_size: int = 512
