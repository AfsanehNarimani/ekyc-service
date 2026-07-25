from fastapi.testclient import TestClient

from api.authentication_dependencies import (
    get_keycloak_jwt_validator,
    authenticate_service,
)
from app import app
from authentication.authenticated_service import AuthenticatedService
from authentication.exceptions import (
    AuthenticationConfigurationError,
)
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from api.dependencies import (
    get_video_verification_orchestrator_service,
)
from authentication.keycloak_jwt_validator import (
    KeycloakJWTValidator,
)


class FakeValidator:
    def validate(
        self,
        token: str,
    ) -> AuthenticatedService:
        return AuthenticatedService(
            client_id="mobile-bank-backend",
        )


class FakeSigningKeyProvider:
    def __init__(
        self,
        public_key: Any,
    ) -> None:
        self.public_key = public_key

    def get_signing_key(
        self,
        token: str,
    ) -> Any:
        return self.public_key


class FakeMissingMediaOrchestrator:
    def verify(
        self,
        correlation_id,
        challenge_id,
        expected_actions,
        national_id_image_uri,
        video_uri,
        face_match_level,
        liveness_level,
    ):
        raise FileNotFoundError(video_uri)


def test_verify_video_rejects_request_without_bearer_token():
    app.dependency_overrides[get_keycloak_jwt_validator] = lambda: FakeValidator()

    app.dependency_overrides.pop(
        authenticate_service,
        None,
    )

    client = TestClient(app)

    response = client.post(
        "/api/verify-video",
        json=make_request_body("trace-auth-001"),
    )

    app.dependency_overrides.clear()

    assert response.status_code == 401

    body = response.json()

    assert body["ok"] is False
    assert body["correlation_id"] == "trace-auth-001"
    assert body["processing_time_ms"] == 0.0
    assert body["error"]["code"] == ("service_authentication_failed")
    assert body["error"]["message"] == ("Service authentication failed.")
    assert body["error"]["details"] == {}


def make_request_body(
    correlation_id: str,
) -> dict:
    return {
        "correlation_id": correlation_id,
        "challenge_id": "challenge-auth-001",
        "expected_actions": [
            "center",
            "left",
            "right",
            "up",
            "down",
        ],
        "national_id_image_uri": "/tmp/id.jpg",
        "video_uri": "/tmp/video.mp4",
        "face_match_level": 2,
        "liveness_level": 2,
    }


def test_verify_video_returns_503_when_keycloak_is_not_configured():
    app.dependency_overrides.pop(
        authenticate_service,
        None,
    )

    def raise_configuration_error():
        raise AuthenticationConfigurationError("keycloak_issuer is not configured.")

    app.dependency_overrides[get_keycloak_jwt_validator] = raise_configuration_error

    client = TestClient(app)

    response = client.post(
        "/api/verify-video",
        headers={
            "Authorization": "Bearer test-token",
        },
        json=make_request_body("trace-auth-config-001"),
    )

    app.dependency_overrides.clear()

    assert response.status_code == 503

    body = response.json()

    assert body["ok"] is False
    assert body["correlation_id"] == "trace-auth-config-001"
    assert body["error"]["code"] == ("service_authentication_unavailable")
    assert body["error"]["message"] == ("Service authentication is unavailable.")
    assert "keycloak_issuer" not in body["error"]["message"]


def test_verify_video_accepts_valid_signed_jwt():
    app.dependency_overrides.pop(
        authenticate_service,
        None,
    )

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    issuer = "https://keycloak.example/realms/mobile-bank"
    now = datetime.now(timezone.utc)

    token = jwt.encode(
        payload={
            "iss": issuer,
            "iat": now,
            "exp": now + timedelta(minutes=5),
            "azp": "mobile-bank-backend",
        },
        key=private_key,
        algorithm="RS256",
        headers={
            "kid": "test-key-id",
        },
    )

    validator = KeycloakJWTValidator(
        signing_key_provider=FakeSigningKeyProvider(private_key.public_key()),
        issuer=issuer,
        allowed_client_id="mobile-bank-backend",
        client_id_claim="azp",
        algorithm="RS256",
        clock_skew_seconds=0,
    )

    app.dependency_overrides[get_keycloak_jwt_validator] = lambda: validator
    app.dependency_overrides[get_video_verification_orchestrator_service] = (
        lambda: FakeMissingMediaOrchestrator()
    )

    client = TestClient(app)

    response = client.post(
        "/api/verify-video",
        headers={
            "Authorization": f"Bearer {token}",
        },
        json=make_request_body("trace-valid-jwt-001"),
    )

    app.dependency_overrides.clear()

    assert response.status_code == 404

    body = response.json()

    assert body["ok"] is False
    assert body["correlation_id"] == "trace-valid-jwt-001"
    assert body["error"]["code"] == "media_file_not_found"
