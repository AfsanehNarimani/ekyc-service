import numpy as np
from PIL import Image

from core.face_detection import FaceDetection
from ai_modules.face_verification.infrastructure.preprocessing.face_aligner import FaceAligner


def test_face_aligner_returns_112x112_image():
    image = Image.new(
        "RGB",
        (200, 200),
        color=(128, 128, 128),
    )

    detection = FaceDetection(
        bbox=(50.0, 50.0, 150.0, 150.0),
        landmarks=np.array(
            [
                [70.0, 80.0],
                [130.0, 80.0],
                [100.0, 110.0],
                [80.0, 140.0],
                [120.0, 140.0],
            ],
            dtype=np.float32,
        ),
        score=0.99,
    )

    aligned = FaceAligner().align(
        image=image,
        detection=detection,
    )

    assert aligned.size == (112, 112)
