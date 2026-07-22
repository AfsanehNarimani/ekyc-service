from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from authentication.exceptions import ServiceAuthenticationError
from authentication.keycloak_jwt_validator import KeycloakJWTValidator

ISSUER = "https://keycloak.example/realms/mobile-bank"
ALLOWED_CLIENT_ID = "mobile-bank-backend"


class FakeSigningKeyProvider:
    def __init__(
        self,
        signing_key: Any,
    ) -> None:
        self.signing_key = signing_key
        self.received_tokens: list[str] = []

    def get_signing_key(
        self,
        token: str,
    ) -> Any:
        self.received_tokens.append(token)
        return self.signing_key


@pytest.fixture
def private_key():
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )


def make_token(
    private_key,
    *,
    issuer: str = ISSUER,
    client_id: str = ALLOWED_CLIENT_ID,
    expires_delta: timedelta = timedelta(minutes=5),
    algorithm: str = "RS256",
    include_client_id: bool = True,
) -> str:
    now = datetime.now(timezone.utc)

    claims = {
        "iss": issuer,
        "iat": now,
        "exp": now + expires_delta,
    }

    if include_client_id:
        claims["azp"] = client_id

    return jwt.encode(
        payload=claims,
        key=private_key,
        algorithm=algorithm,
    )


def make_validator(
    public_key,
) -> KeycloakJWTValidator:
    return KeycloakJWTValidator(
        signing_key_provider=FakeSigningKeyProvider(public_key),
        issuer=ISSUER,
        allowed_client_id=ALLOWED_CLIENT_ID,
        client_id_claim="azp",
        algorithm="RS256",
        clock_skew_seconds=0,
    )


def test_keycloak_jwt_validator_accepts_valid_allowed_client(
    private_key,
):
    token = make_token(private_key)

    validator = make_validator(
        private_key.public_key(),
    )

    result = validator.validate(token)

    assert result.client_id == ALLOWED_CLIENT_ID


def test_keycloak_jwt_validator_rejects_unallowed_client(
    private_key,
):
    token = make_token(
        private_key,
        client_id="unallowed-service",
    )

    validator = make_validator(
        private_key.public_key(),
    )

    with pytest.raises(
        ServiceAuthenticationError,
        match="Service authentication failed",
    ):
        validator.validate(token)


def test_keycloak_jwt_validator_rejects_expired_token(
    private_key,
):
    token = make_token(
        private_key,
        expires_delta=timedelta(seconds=-1),
    )

    validator = make_validator(
        private_key.public_key(),
    )

    with pytest.raises(ServiceAuthenticationError):
        validator.validate(token)


def test_keycloak_jwt_validator_rejects_invalid_issuer(
    private_key,
):
    token = make_token(
        private_key,
        issuer="https://untrusted.example/realms/fake",
    )

    validator = make_validator(
        private_key.public_key(),
    )

    with pytest.raises(ServiceAuthenticationError):
        validator.validate(token)


def test_keycloak_jwt_validator_rejects_invalid_signature(
    private_key,
):
    other_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    token = make_token(other_private_key)

    validator = make_validator(
        private_key.public_key(),
    )

    with pytest.raises(ServiceAuthenticationError):
        validator.validate(token)


def test_keycloak_jwt_validator_rejects_missing_client_id_claim(
    private_key,
):
    token = make_token(
        private_key,
        include_client_id=False,
    )

    validator = make_validator(
        private_key.public_key(),
    )

    with pytest.raises(ServiceAuthenticationError):
        validator.validate(token)


def test_keycloak_jwt_validator_rejects_unexpected_algorithm(
    private_key,
):
    now = datetime.now(timezone.utc)

    token = jwt.encode(
        payload={
            "iss": ISSUER,
            "iat": now,
            "exp": now + timedelta(minutes=5),
            "azp": ALLOWED_CLIENT_ID,
        },
        key="test-only-secret-at-least-32-bytes",
        algorithm="HS256",
    )

    validator = make_validator(
        private_key.public_key(),
    )

    with pytest.raises(ServiceAuthenticationError):
        validator.validate(token)


def test_keycloak_jwt_validator_rejects_empty_token(
    private_key,
):
    validator = make_validator(
        private_key.public_key(),
    )

    with pytest.raises(ServiceAuthenticationError):
        validator.validate("")
