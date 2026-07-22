from unittest.mock import Mock

import pytest

from authentication.signing_key_provider import (
    KeycloakSigningKeyProvider,
)


def test_keycloak_signing_key_provider_configures_jwks_cache(
    monkeypatch,
):
    jwks_client = Mock()
    jwks_client_class = Mock(
        return_value=jwks_client,
    )

    monkeypatch.setattr(
        "authentication.signing_key_provider.PyJWKClient",
        jwks_client_class,
    )

    expected_signing_key = object()
    jwks_client.get_signing_key_from_jwt.return_value = expected_signing_key

    provider = KeycloakSigningKeyProvider(
        jwks_url=(
            "https://keycloak.example/realms/"
            "mobile-bank/protocol/openid-connect/certs"
        ),
        cache_seconds=300,
        timeout_seconds=5.0,
    )

    result = provider.get_signing_key("header.payload.signature")

    assert result is expected_signing_key

    jwks_client_class.assert_called_once_with(
        uri=(
            "https://keycloak.example/realms/"
            "mobile-bank/protocol/openid-connect/certs"
        ),
        cache_jwk_set=True,
        lifespan=300,
        cache_keys=True,
        timeout=5.0,
    )

    jwks_client.get_signing_key_from_jwt.assert_called_once_with(
        "header.payload.signature"
    )


@pytest.mark.parametrize(
    (
        "jwks_url",
        "cache_seconds",
        "timeout_seconds",
        "expected_message",
    ),
    [
        (
            "",
            300,
            5.0,
            "jwks_url must not be empty",
        ),
        (
            "https://keycloak.example/certs",
            0,
            5.0,
            "cache_seconds must be greater than zero",
        ),
        (
            "https://keycloak.example/certs",
            300,
            0.0,
            "timeout_seconds must be greater than zero",
        ),
    ],
)
def test_keycloak_signing_key_provider_rejects_invalid_configuration(
    jwks_url,
    cache_seconds,
    timeout_seconds,
    expected_message,
):
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        KeycloakSigningKeyProvider(
            jwks_url=jwks_url,
            cache_seconds=cache_seconds,
            timeout_seconds=timeout_seconds,
        )
