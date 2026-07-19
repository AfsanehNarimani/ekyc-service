from pathlib import Path

import numpy as np

from core.face_embedding import FaceEmbedding


class EmbeddingStore:
    def __init__(
        self,
        root_dir: Path,
    ) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        person_id: str,
        embedding: FaceEmbedding,
    ) -> Path:
        path = self.root_dir / f"{person_id}.npy"

        np.save(
            path,
            embedding.vector.astype(np.float32),
        )

        return path

    def load(
        self,
        person_id: str,
        model_name: str,
    ) -> FaceEmbedding:
        path = self.root_dir / f"{person_id}.npy"

        if not path.exists():
            raise FileNotFoundError(path)

        vector = np.load(path)

        return FaceEmbedding(
            vector=vector.astype(np.float32),
            model_name=model_name,
        )

    def exists(
        self,
        person_id: str,
    ) -> bool:
        return (self.root_dir / f"{person_id}.npy").exists()
