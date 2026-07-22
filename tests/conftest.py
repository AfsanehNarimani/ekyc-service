import pytest

from api.authentication_dependencies import authenticate_service
from app import app
from authentication.authenticated_service import AuthenticatedService


@pytest.fixture(autouse=True)
def bypass_service_authentication():
    app.dependency_overrides[authenticate_service] = lambda: (
        AuthenticatedService(
            client_id="test-backend",
        )
    )

    yield

    app.dependency_overrides.pop(
        authenticate_service,
        None,
    )
