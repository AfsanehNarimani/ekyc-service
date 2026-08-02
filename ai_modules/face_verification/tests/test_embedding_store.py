import numpy as np

from ai_modules.face_verification.domain.face_embedding import FaceEmbedding
from storage.embedding_store import EmbeddingStore


def test_embedding_store_saves_and_loads_embedding(tmp_path):
    store = EmbeddingStore(
        root_dir=tmp_path,
    )

    embedding = FaceEmbedding(
        vector=np.ones(
            512,
            dtype=np.float32,
        ),
        model_name="edgeface",
    )

    path = store.save(
        person_id="user123",
        embedding=embedding,
    )

    loaded = store.load(
        person_id="user123",
        model_name="edgeface",
    )

    assert path.exists()
    assert loaded.model_name == "edgeface"
    assert loaded.vector.shape == (512,)
    assert loaded.vector.dtype == np.float32
    assert np.allclose(
        loaded.vector,
        embedding.vector,
    )
