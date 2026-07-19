from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.dependencies import get_readiness_service
from app import app


class FakeReadyReadinessService:
    def check(self):
        return SimpleNamespace(
            ready=True,
            checks=[
                SimpleNamespace(
                    name="default_face_model",
                    passed=True,
                    message="model file exists",
                )
            ],
        )


class FakeNotReadyReadinessService:
    def check(self):
        return SimpleNamespace(
            ready=False,
            checks=[
                SimpleNamespace(
                    name="default_face_model",
                    passed=False,
                    message="model file is missing",
                )
            ],
        )


def test_ready_endpoint_returns_200_when_service_is_ready():
    app.dependency_overrides[get_readiness_service] = (
        lambda: FakeReadyReadinessService()
    )

    client = TestClient(app)

    response = client.get("/ready")

    app.dependency_overrides.clear()

    assert response.status_code == 200

    body = response.json()

    assert body["ok"] is True
    assert body["ready"] is True
    assert body["checks"][0]["passed"] is True


def test_ready_endpoint_returns_503_when_service_is_not_ready():
    app.dependency_overrides[get_readiness_service] = (
        lambda: FakeNotReadyReadinessService()
    )

    client = TestClient(app)

    response = client.get("/ready")

    app.dependency_overrides.clear()

    assert response.status_code == 503

    body = response.json()

    assert body["ok"] is False
    assert body["ready"] is False
    assert body["checks"][0]["passed"] is False
