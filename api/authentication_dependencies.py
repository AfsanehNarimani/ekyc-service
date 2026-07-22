from functools import lru_cache

from fastapi import Depends
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from authentication.authenticated_service import AuthenticatedService
from authentication.exceptions import (
    AuthenticationConfigurationError,
    ServiceAuthenticationError,
)
from authentication.keycloak_jwt_validator import KeycloakJWTValidator
from authentication.signing_key_provider import (
    KeycloakSigningKeyProvider,
)
from config import settings

bearer_scheme = HTTPBearer(
    auto_error=False,
)


def _required_setting(
    value: str | None,
    name: str,
) -> str:
    if value is None or not value.strip():
        raise AuthenticationConfigurationError(f"{name} is not configured.")

    return value


@lru_cache
def get_keycloak_jwt_validator() -> KeycloakJWTValidator | None:
    if not settings.keycloak_auth_enabled:
        return None

    issuer = _required_setting(
        settings.keycloak_issuer,
        "keycloak_issuer",
    )
    jwks_url = _required_setting(
        settings.keycloak_jwks_url,
        "keycloak_jwks_url",
    )
    allowed_client_id = _required_setting(
        settings.keycloak_allowed_client_id,
        "keycloak_allowed_client_id",
    )

    signing_key_provider = KeycloakSigningKeyProvider(
        jwks_url=jwks_url,
        cache_seconds=settings.keycloak_jwks_cache_seconds,
        timeout_seconds=settings.keycloak_http_timeout_seconds,
    )

    return KeycloakJWTValidator(
        signing_key_provider=signing_key_provider,
        issuer=issuer,
        allowed_client_id=allowed_client_id,
        client_id_claim=settings.keycloak_client_id_claim,
        algorithm=settings.keycloak_algorithm,
        clock_skew_seconds=settings.keycloak_clock_skew_seconds,
    )


def authenticate_service(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    validator: KeycloakJWTValidator | None = Depends(get_keycloak_jwt_validator),
) -> AuthenticatedService | None:
    if validator is None:
        return None

    if credentials is None:
        raise ServiceAuthenticationError("Service authentication failed.")

    return validator.validate(
        credentials.credentials,
    )
