from pathlib import Path

from face import insightface_detector
from face.insightface_detector import (
    InsightFaceDetector,
)


class FakeFaceAnalysis:
    received_arguments = None
    prepared_arguments = None

    def __init__(
        self,
        **kwargs,
    ) -> None:
        FakeFaceAnalysis.received_arguments = kwargs

    def prepare(
        self,
        **kwargs,
    ) -> None:
        FakeFaceAnalysis.prepared_arguments = kwargs


def test_insightface_detector_loads_only_detection_model(
    monkeypatch,
):
    monkeypatch.setattr(
        insightface_detector,
        "FaceAnalysis",
        FakeFaceAnalysis,
    )

    InsightFaceDetector(
        det_thresh=0.6,
        det_size=(
            320,
            320,
        ),
        model_name="buffalo_l",
        root_dir=Path("/models/insightface"),
        providers=[
            "CPUExecutionProvider",
        ],
    )

    assert FakeFaceAnalysis.received_arguments == {
        "name": "buffalo_l",
        "root": "/models/insightface",
        "allowed_modules": [
            "detection",
        ],
        "providers": [
            "CPUExecutionProvider",
        ],
    }

    assert FakeFaceAnalysis.prepared_arguments == {
        "ctx_id": 0,
        "det_thresh": 0.6,
        "det_size": (
            320,
            320,
        ),
    }
