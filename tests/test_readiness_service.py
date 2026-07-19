from pathlib import Path

from config import settings
from services.readiness_service import ReadinessService


def test_readiness_service_passes_when_required_paths_exist(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.write_text("fake model")

    monkeypatch.setattr(settings, "default_face_model", "edgeface")
    monkeypatch.setattr(settings, "edgeface_model_path", model_path)
    monkeypatch.setattr(settings, "media_temp_dir", tmp_path / "media")
    monkeypatch.setattr(settings, "reference_dir", tmp_path / "references")
    monkeypatch.setattr(settings, "embedding_dir", tmp_path / "embeddings")

    result = ReadinessService().check()

    assert result.ready is True
    assert all(check.passed for check in result.checks)


def test_readiness_service_fails_when_default_model_is_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(settings, "default_face_model", "edgeface")
    monkeypatch.setattr(settings, "edgeface_model_path", Path("/tmp/missing.onnx"))
    monkeypatch.setattr(settings, "media_temp_dir", tmp_path / "media")
    monkeypatch.setattr(settings, "reference_dir", tmp_path / "references")
    monkeypatch.setattr(settings, "embedding_dir", tmp_path / "embeddings")

    result = ReadinessService().check()

    assert result.ready is False
    assert any(
        check.name == "default_face_model" and check.passed is False
        for check in result.checks
    )
