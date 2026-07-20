from pathlib import Path

from config import settings
from services.readiness_service import ReadinessService
import pytest


@pytest.fixture(autouse=True)
def configure_passive_liveness_settings(monkeypatch):
    monkeypatch.setattr(
        settings,
        "passive_liveness_backend",
        "stub",
    )
    monkeypatch.setattr(
        settings,
        "passive_liveness_decision_mode",
        "report_only",
    )


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


def test_readiness_service_passes_when_passive_model_exists(
    tmp_path,
    monkeypatch,
):
    face_model_path = tmp_path / "face_model.onnx"
    face_model_path.write_text("fake face model")

    passive_v1se_model_path = tmp_path / "passive_v1se_model.onnx"
    passive_v1se_model_path.write_text("fake passive v1se model")

    passive_model_path = tmp_path / "passive_model.onnx"
    passive_model_path.write_text("fake passive model")

    monkeypatch.setattr(settings, "default_face_model", "edgeface")
    monkeypatch.setattr(settings, "edgeface_model_path", face_model_path)
    monkeypatch.setattr(settings, "passive_liveness_backend", "onnx")
    monkeypatch.setattr(
        settings,
        "passive_liveness_model_path",
        passive_model_path,
    )
    monkeypatch.setattr(settings, "media_temp_dir", tmp_path / "media")
    monkeypatch.setattr(settings, "reference_dir", tmp_path / "references")
    monkeypatch.setattr(settings, "embedding_dir", tmp_path / "embeddings")

    monkeypatch.setattr(
        settings,
        "passive_liveness_v1se_model_path",
        passive_v1se_model_path,
    )

    result = ReadinessService().check()

    assert result.ready is True
    assert any(
        check.name == "passive_liveness_model" and check.passed is True
        for check in result.checks
    )


def test_readiness_service_fails_when_passive_model_is_missing(
    tmp_path,
    monkeypatch,
):
    face_model_path = tmp_path / "face_model.onnx"
    face_model_path.write_text("fake face model")

    passive_v1se_model_path = tmp_path / "passive_v1se_model.onnx"
    passive_v1se_model_path.write_text("fake passive v1se model")

    monkeypatch.setattr(settings, "default_face_model", "edgeface")
    monkeypatch.setattr(settings, "edgeface_model_path", face_model_path)
    monkeypatch.setattr(settings, "passive_liveness_backend", "onnx")
    monkeypatch.setattr(
        settings,
        "passive_liveness_model_path",
        tmp_path / "missing_passive_model.onnx",
    )
    monkeypatch.setattr(settings, "media_temp_dir", tmp_path / "media")
    monkeypatch.setattr(settings, "reference_dir", tmp_path / "references")
    monkeypatch.setattr(settings, "embedding_dir", tmp_path / "embeddings")

    monkeypatch.setattr(
        settings,
        "passive_liveness_v1se_model_path",
        passive_v1se_model_path,
    )

    result = ReadinessService().check()

    assert result.ready is False
    assert any(
        check.name == "passive_liveness_model" and check.passed is False
        for check in result.checks
    )


def test_readiness_service_fails_for_invalid_decision_mode(
    tmp_path,
    monkeypatch,
):
    face_model_path = tmp_path / "face_model.onnx"
    face_model_path.write_text("fake face model")

    monkeypatch.setattr(settings, "default_face_model", "edgeface")
    monkeypatch.setattr(settings, "edgeface_model_path", face_model_path)
    monkeypatch.setattr(
        settings,
        "passive_liveness_decision_mode",
        "invalid",
    )
    monkeypatch.setattr(settings, "media_temp_dir", tmp_path / "media")
    monkeypatch.setattr(settings, "reference_dir", tmp_path / "references")
    monkeypatch.setattr(settings, "embedding_dir", tmp_path / "embeddings")

    result = ReadinessService().check()

    assert result.ready is False
    assert any(
        check.name == "passive_liveness_decision_mode" and check.passed is False
        for check in result.checks
    )
