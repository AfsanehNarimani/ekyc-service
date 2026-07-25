from pathlib import Path

from config import settings
from services.readiness_service import ReadinessService
import pytest


@pytest.fixture(autouse=True)
def configure_readiness_settings(monkeypatch):
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
    monkeypatch.setattr(
        settings,
        "keycloak_auth_enabled",
        False,
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


def test_readiness_service_passes_when_keycloak_is_configured(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "model.onnx"
    model_path.write_text("fake model")

    monkeypatch.setattr(
        settings,
        "default_face_model",
        "edgeface",
    )
    monkeypatch.setattr(
        settings,
        "edgeface_model_path",
        model_path,
    )
    monkeypatch.setattr(
        settings,
        "media_temp_dir",
        tmp_path / "media",
    )
    monkeypatch.setattr(
        settings,
        "reference_dir",
        tmp_path / "references",
    )
    monkeypatch.setattr(
        settings,
        "embedding_dir",
        tmp_path / "embeddings",
    )

    monkeypatch.setattr(
        settings,
        "keycloak_auth_enabled",
        True,
    )
    monkeypatch.setattr(
        settings,
        "keycloak_issuer",
        "https://keycloak.example/realms/mobile-bank",
    )
    monkeypatch.setattr(
        settings,
        "keycloak_jwks_url",
        (
            "https://keycloak.example/realms/mobile-bank/"
            "protocol/openid-connect/certs"
        ),
    )
    monkeypatch.setattr(
        settings,
        "keycloak_allowed_client_id",
        "mobile-bank-backend",
    )
    monkeypatch.setattr(
        settings,
        "keycloak_client_id_claim",
        "azp",
    )
    monkeypatch.setattr(
        settings,
        "keycloak_algorithm",
        "RS256",
    )

    result = ReadinessService().check()

    assert result.ready is True
    assert any(
        check.name == "keycloak_authentication" and check.passed is True
        for check in result.checks
    )


def test_readiness_service_fails_when_keycloak_settings_are_missing(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "model.onnx"
    model_path.write_text("fake model")

    monkeypatch.setattr(
        settings,
        "default_face_model",
        "edgeface",
    )
    monkeypatch.setattr(
        settings,
        "edgeface_model_path",
        model_path,
    )
    monkeypatch.setattr(
        settings,
        "media_temp_dir",
        tmp_path / "media",
    )
    monkeypatch.setattr(
        settings,
        "reference_dir",
        tmp_path / "references",
    )
    monkeypatch.setattr(
        settings,
        "embedding_dir",
        tmp_path / "embeddings",
    )

    monkeypatch.setattr(
        settings,
        "keycloak_auth_enabled",
        True,
    )
    monkeypatch.setattr(
        settings,
        "keycloak_issuer",
        None,
    )
    monkeypatch.setattr(
        settings,
        "keycloak_jwks_url",
        None,
    )
    monkeypatch.setattr(
        settings,
        "keycloak_allowed_client_id",
        None,
    )

    result = ReadinessService().check()

    assert result.ready is False

    keycloak_check = next(
        check for check in result.checks if check.name == "keycloak_authentication"
    )

    assert keycloak_check.passed is False
    assert "keycloak_issuer" in keycloak_check.message
    assert "keycloak_jwks_url" in keycloak_check.message
    assert "keycloak_allowed_client_id" in keycloak_check.message


def test_readiness_service_fails_for_invalid_verification_capacity(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "model.onnx"
    model_path.write_text("fake model")

    monkeypatch.setattr(
        settings,
        "default_face_model",
        "edgeface",
    )
    monkeypatch.setattr(
        settings,
        "edgeface_model_path",
        model_path,
    )
    monkeypatch.setattr(
        settings,
        "media_temp_dir",
        tmp_path / "media",
    )
    monkeypatch.setattr(
        settings,
        "reference_dir",
        tmp_path / "references",
    )
    monkeypatch.setattr(
        settings,
        "embedding_dir",
        tmp_path / "embeddings",
    )
    monkeypatch.setattr(
        settings,
        "max_concurrent_verifications",
        0,
    )

    result = ReadinessService().check()

    assert result.ready is False
    assert any(
        check.name == "verification_capacity" and check.passed is False
        for check in result.checks
    )


def test_readiness_service_fails_for_invalid_retry_after(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "model.onnx"
    model_path.write_text("fake model")

    monkeypatch.setattr(
        settings,
        "default_face_model",
        "edgeface",
    )
    monkeypatch.setattr(
        settings,
        "edgeface_model_path",
        model_path,
    )
    monkeypatch.setattr(
        settings,
        "media_temp_dir",
        tmp_path / "media",
    )
    monkeypatch.setattr(
        settings,
        "reference_dir",
        tmp_path / "references",
    )
    monkeypatch.setattr(
        settings,
        "embedding_dir",
        tmp_path / "embeddings",
    )
    monkeypatch.setattr(
        settings,
        "capacity_retry_after_seconds",
        0,
    )

    result = ReadinessService().check()

    assert result.ready is False
    assert any(
        check.name == "verification_capacity" and check.passed is False
        for check in result.checks
    )
