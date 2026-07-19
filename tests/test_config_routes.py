from fastapi.testclient import TestClient

from app import app


def test_verification_config_endpoint_returns_config():
    client = TestClient(app)

    response = client.get("/api/verification-config")

    assert response.status_code == 200

    body = response.json()

    assert body["default_face_model"] == "edgeface"
    assert body["supported_actions"] == [
        "center",
        "left",
        "right",
        "up",
        "down",
    ]
    assert body["face_match_levels"]["2"]["name"] == "normal"
    assert body["liveness_levels"]["2"]["name"] == "normal"
