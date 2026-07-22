from fastapi.testclient import TestClient

from api.authentication_dependencies import (
    authenticate_service,
    get_keycloak_jwt_validator,
)
from app import app
from authentication.authenticated_service import AuthenticatedService


class FakeValidator:
    def validate(
        self,
        token: str,
    ) -> AuthenticatedService:
        return AuthenticatedService(
            client_id="mobile-bank-backend",
        )


def test_verification_config_rejects_missing_bearer_token():
    app.dependency_overrides.pop(
        authenticate_service,
        None,
    )
    app.dependency_overrides[get_keycloak_jwt_validator] = (
        lambda: FakeValidator()
    )

    client = TestClient(app)

    response = client.get(
        "/api/verification-config"
    )

    app.dependency_overrides.clear()

    assert response.status_code == 401

    body = response.json()

    assert body["ok"] is False
    assert body["trace_id"] is None
    assert body["error"]["code"] == (
        "service_authentication_failed"
    )
    assert body["error"]["message"] == (
        "Service authentication failed."
    )
    assert body["error"]["details"] == {}
