from typing import Any, Protocol

from jwt import PyJWKClient


class SigningKeyProvider(Protocol):
    def get_signing_key(
        self,
        token: str,
    ) -> Any:
        pass


class KeycloakSigningKeyProvider:
    def __init__(
        self,
        jwks_url: str,
        cache_seconds: int,
        timeout_seconds: float,
    ) -> None:
        if not jwks_url:
            raise ValueError("jwks_url must not be empty.")

        if cache_seconds <= 0:
            raise ValueError("cache_seconds must be greater than zero.")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        self.jwks_client = PyJWKClient(
            uri=jwks_url,
            cache_jwk_set=True,
            lifespan=cache_seconds,
            cache_keys=True,
            timeout=timeout_seconds,
        )

    def get_signing_key(
        self,
        token: str,
    ) -> Any:
        return self.jwks_client.get_signing_key_from_jwt(token)
