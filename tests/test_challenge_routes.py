from fastapi.testclient import TestClient

from app import app


def test_create_challenge_endpoint_returns_hardcoded_actions():
    client = TestClient(app)

    response = client.post("/api/challenges")

    assert response.status_code == 200

    body = response.json()

    assert body["ok"] is True
    assert body["challenge_id"]
    assert body["actions"] == [
        "center",
        "left",
        "right",
        "up",
        "down",
    ]
    assert body["expires_in_seconds"] > 0
