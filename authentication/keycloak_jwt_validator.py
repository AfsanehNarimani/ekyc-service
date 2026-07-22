from __future__ import annotations

from typing import Any

import jwt
from jwt.exceptions import PyJWKClientError, PyJWTError

from authentication.authenticated_service import AuthenticatedService
from authentication.exceptions import ServiceAuthenticationError
from authentication.signing_key_provider import SigningKeyProvider


class KeycloakJWTValidator:
    def __init__(
        self,
        signing_key_provider: SigningKeyProvider,
        issuer: str,
        allowed_client_id: str,
        client_id_claim: str = "azp",
        algorithm: str = "RS256",
        clock_skew_seconds: int = 10,
    ) -> None:
        if not issuer:
            raise ValueError("issuer must not be empty.")

        if not allowed_client_id:
            raise ValueError("allowed_client_id must not be empty.")

        if not client_id_claim:
            raise ValueError("client_id_claim must not be empty.")

        if not algorithm:
            raise ValueError("algorithm must not be empty.")

        if clock_skew_seconds < 0:
            raise ValueError("clock_skew_seconds must not be negative.")

        self.signing_key_provider = signing_key_provider
        self.issuer = issuer
        self.allowed_client_id = allowed_client_id
        self.client_id_claim = client_id_claim
        self.algorithm = algorithm
        self.clock_skew_seconds = clock_skew_seconds

    def validate(
        self,
        token: str,
    ) -> AuthenticatedService:
        if not token:
            raise ServiceAuthenticationError("Service authentication failed.")

        try:
            signing_key = self.signing_key_provider.get_signing_key(token)

            claims: dict[str, Any] = jwt.decode(
                jwt=token,
                key=signing_key,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                leeway=self.clock_skew_seconds,
                options={
                    "verify_aud": False,
                    "require": [
                        "exp",
                        "iss",
                        self.client_id_claim,
                    ],
                },
            )

        except (
            PyJWTError,
            PyJWKClientError,
        ) as error:
            raise ServiceAuthenticationError(
                "Service authentication failed."
            ) from error

        client_id = claims.get(self.client_id_claim)

        if not isinstance(client_id, str) or client_id != self.allowed_client_id:
            raise ServiceAuthenticationError("Service authentication failed.")

        return AuthenticatedService(
            client_id=client_id,
        )
