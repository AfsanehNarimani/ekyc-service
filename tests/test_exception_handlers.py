from fastapi.testclient import TestClient

from app import app


def test_request_validation_error_returns_standard_error_response():
    client = TestClient(app)

    response = client.post(
        "/api/verify-video",
        json={
            "correlation_id": "trace-validation-1",
            "challenge_id": "backend-challenge-123",
            "expected_actions": [
                "center",
                "left",
                "right",
                "up",
                "down",
            ],
            "national_id_image_uri": "/tmp/id.jpg",
            "video_uri": "/tmp/video.mp4",
            "face_match_level": 9,
            "liveness_level": 2,
        },
    )

    assert response.status_code == 422

    body = response.json()

    assert body["ok"] is False
    assert body["correlation_id"] == "trace-validation-1"
    assert body["processing_time_ms"] == 0.0
    assert body["error"]["code"] == "request_validation_failed"
    assert body["error"]["message"] == "Request validation failed."
    assert "errors" in body["error"]["details"]

    first_error = body["error"]["details"]["errors"][0]

    assert "input" not in first_error
    assert first_error["loc"] == [
        "body",
        "face_match_level",
    ]


def test_request_validation_error_without_trace_id_returns_null_correlation_id():
    client = TestClient(app)

    response = client.post(
        "/api/verify-video",
        json={
            "challenge_id": "backend-challenge-123",
            "expected_actions": [
                "center",
                "left",
                "right",
                "up",
                "down",
            ],
            "national_id_image_uri": "/tmp/id.jpg",
            "video_uri": "/tmp/video.mp4",
            "face_match_level": 9,
            "liveness_level": 2,
        },
    )

    assert response.status_code == 422

    body = response.json()

    assert body["ok"] is False
    assert body["correlation_id"] is None
    assert body["error"]["code"] == "request_validation_failed"
