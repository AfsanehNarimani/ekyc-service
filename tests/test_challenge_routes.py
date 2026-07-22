from fastapi.testclient import TestClient

from app import app


def test_challenge_endpoint_is_not_published():
    client = TestClient(app)

    response = client.post("/api/challenges")

    assert response.status_code == 404
