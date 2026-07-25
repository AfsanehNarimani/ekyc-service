from fastapi.testclient import TestClient

from api.capacity_dependencies import (
    get_verification_capacity_limiter,
)
from app import app
from services.verification_capacity_limiter import (
    VerificationCapacityLimiter,
)


def make_request_body() -> dict:
    return {
        "trace_id": "trace-capacity-001",
        "challenge_id": "challenge-capacity-001",
        "expected_actions": [
            "center",
            "left",
            "right",
            "up",
            "down",
        ],
        "national_id_image_uri": "/tmp/id.jpg",
        "video_uri": "/tmp/video.mp4",
        "face_match_level": 2,
        "liveness_level": 2,
    }


def test_verify_video_returns_503_when_capacity_is_exhausted():
    limiter = VerificationCapacityLimiter(max_concurrent_verifications=1)

    app.dependency_overrides[get_verification_capacity_limiter] = lambda: limiter

    client = TestClient(app)

    with limiter.acquire():
        response = client.post(
            "/api/verify-video",
            json=make_request_body(),
        )

    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "3"

    body = response.json()

    assert body["ok"] is False
    assert body["trace_id"] == "trace-capacity-001"
    assert body["processing_time_ms"] == 0.0
    assert body["error"]["code"] == ("verification_capacity_exceeded")
    assert body["error"]["message"] == (
        "Verification capacity is temporarily unavailable."
    )
    assert body["error"]["details"] == {}
