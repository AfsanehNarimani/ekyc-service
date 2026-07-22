from fastapi.security import HTTPAuthorizationCredentials
import pytest

from api.authentication_dependencies import authenticate_service
from authentication.authenticated_service import AuthenticatedService
from authentication.exceptions import ServiceAuthenticationError


class FakeAcceptingValidator:
    def validate(
        self,
        token: str,
    ) -> AuthenticatedService:
        assert token == "valid-test-token"

        return AuthenticatedService(
            client_id="mobile-bank-backend",
        )


class FakeRejectingValidator:
    def validate(
        self,
        token: str,
    ) -> AuthenticatedService:
        raise ServiceAuthenticationError("Service authentication failed.")


def test_authenticate_service_accepts_valid_bearer_token():
    result = authenticate_service(
        credentials=HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="valid-test-token",
        ),
        validator=FakeAcceptingValidator(),
    )

    assert result is not None
    assert result.client_id == "mobile-bank-backend"


def test_authenticate_service_rejects_missing_credentials():
    with pytest.raises(ServiceAuthenticationError):
        authenticate_service(
            credentials=None,
            validator=FakeAcceptingValidator(),
        )


def test_authenticate_service_rejects_invalid_token():
    with pytest.raises(ServiceAuthenticationError):
        authenticate_service(
            credentials=HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="invalid-test-token",
            ),
            validator=FakeRejectingValidator(),
        )


def test_authenticate_service_is_disabled_without_validator():
    result = authenticate_service(
        credentials=None,
        validator=None,
    )

    assert result is None
