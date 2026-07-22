from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuthenticatedService:
    client_id: str
