from fastapi.testclient import TestClient

from api.dependencies import get_video_verification_orchestrator_service
from app import app
from core.action_check_result import ActionCheckResult
from core.final_decision import FinalDecision
from core.liveness_action import LivenessAction
from core.passive_liveness_result import PassiveLivenessResult
from core.video_face_match_result import VideoFaceMatchResult
from core.video_liveness_result import VideoLivenessResult
from core.video_verification_result import VideoVerificationResult


class FakeVideoVerificationOrchestratorService:
    def verify(
        self,
        correlation_id: str,
        challenge_id: str,
        expected_actions: list[LivenessAction],
        national_id_image_uri: str,
        video_uri: str,
        face_match_level: int,
        liveness_level: int,
    ) -> VideoVerificationResult:
        return VideoVerificationResult(
            correlation_id=correlation_id,
            challenge_id=challenge_id,
            expected_actions=expected_actions,
            face_match=VideoFaceMatchResult(
                is_match=True,
                score=0.76,
                best_score=0.82,
                threshold=0.55,
                model_name="edgeface",
                processed_frames=10,
                skipped_frames=2,
                frame_scores=[0.76],
            ),
            action_check=ActionCheckResult(
                passed=True,
                required_actions=expected_actions,
                observed_actions=expected_actions,
                compact_actions=expected_actions,
                missing_actions=[],
                processed_frames=10,
                skipped_frames=0,
            ),
            liveness=VideoLivenessResult(
                passed=True,
                strictness_level=liveness_level,
                processed_frames=10,
                skipped_frames=0,
                min_required_frames=5,
                pose_variation_score=30.0,
                min_pose_variation_score=18.0,
                reason="passed_basic_liveness",
            ),
            passive_liveness=PassiveLivenessResult(
                passed=True,
                score=1.0,
                threshold=0.6,
                model_name="stub_passive_liveness",
                processed_frames=0,
                skipped_frames=0,
                reason="stub_passive_liveness_passed",
            ),
            final_decision=FinalDecision(
                approved=True,
                status="approved",
                rejection_reasons=[],
                face_match_passed=True,
                actions_passed=True,
                liveness_passed=True,
                passive_liveness_passed=True,
                passive_liveness_enforced=False,
            ),
            processing_time_ms=123.45,
        )


class FakeFailingVideoVerificationOrchestratorService:
    def verify(
        self,
        correlation_id: str,
        challenge_id,
        expected_actions,
        national_id_image_uri,
        video_uri,
        face_match_level,
        liveness_level,
    ):
        raise FileNotFoundError(video_uri)


class FakeUnexpectedFailingVideoVerificationOrchestratorService:
    def verify(
        self,
        correlation_id: str,
        challenge_id,
        expected_actions,
        national_id_image_uri,
        video_uri,
        face_match_level,
        liveness_level,
    ):
        raise RuntimeError("database password or internal path should not leak")


class FakeInvalidActionVideoVerificationOrchestratorService:
    def verify(
        self,
        correlation_id: str,
        challenge_id,
        expected_actions,
        national_id_image_uri,
        video_uri,
        face_match_level,
        liveness_level,
    ):
        raise ValueError(
            "Invalid expected_actions. Expected exactly: "
            "['center', 'left', 'right', 'up', 'down']"
        )


def test_verify_video_endpoint_returns_final_approved_result():
    app.dependency_overrides[get_video_verification_orchestrator_service] = (
        lambda: FakeVideoVerificationOrchestratorService()
    )

    client = TestClient(app)

    response = client.post(
        "/api/verify-video",
        json={
            "correlation_id": "trace-1",
            "challenge_id": "backend-challenge-123",
            "expected_actions": [
                "center",
                "left",
                "right",
                "up",
                "down",
            ],
            "national_id_image_uri": "s3://bucket/id.jpg",
            "video_uri": "s3://bucket/video.mp4",
            "face_match_level": 2,
            "liveness_level": 2,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200

    body = response.json()

    assert body["ok"] is True
    assert body["correlation_id"] == "trace-1"
    assert body["challenge_id"] == "backend-challenge-123"

    assert body["face_match"]["passed"] is True
    assert body["action_check"]["passed"] is True
    assert body["liveness"]["passed"] is True
    assert body["passive_liveness"]["passed"] is True
    assert body["passive_liveness"]["model_name"] == "stub_passive_liveness"

    assert body["final_decision"]["approved"] is True
    assert body["final_decision"]["status"] == "approved"
    assert body["final_decision"]["rejection_reasons"] == []
    assert body["final_decision"]["face_match_passed"] is True
    assert body["final_decision"]["actions_passed"] is True
    assert body["final_decision"]["liveness_passed"] is True
    assert body["final_decision"]["passive_liveness_passed"] is True
    assert body["processing_time_ms"] == 123.45


def test_verify_video_endpoint_returns_standard_error_response():
    app.dependency_overrides[get_video_verification_orchestrator_service] = (
        lambda: FakeFailingVideoVerificationOrchestratorService()
    )

    client = TestClient(app)

    response = client.post(
        "/api/verify-video",
        json={
            "correlation_id": "trace-error-1",
            "challenge_id": "backend-challenge-123",
            "expected_actions": [
                "center",
                "left",
                "right",
                "up",
                "down",
            ],
            "national_id_image_uri": "/tmp/id.jpg",
            "video_uri": "/tmp/missing.mp4",
            "face_match_level": 2,
            "liveness_level": 2,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 404

    body = response.json()

    assert body["ok"] is False
    assert body["correlation_id"] == "trace-error-1"
    assert body["processing_time_ms"] >= 0
    assert body["error"]["code"] == "media_file_not_found"
    assert "Media file not found" in body["error"]["message"]
    assert body["error"]["details"] == {}


def test_verify_video_endpoint_returns_safe_internal_error_response():
    app.dependency_overrides[get_video_verification_orchestrator_service] = (
        lambda: FakeUnexpectedFailingVideoVerificationOrchestratorService()
    )

    client = TestClient(app)

    response = client.post(
        "/api/verify-video",
        json={
            "correlation_id": "trace-internal-error-1",
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
            "face_match_level": 2,
            "liveness_level": 2,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 500

    body = response.json()

    assert body["ok"] is False
    assert body["correlation_id"] == "trace-internal-error-1"
    assert body["processing_time_ms"] >= 0
    assert body["error"]["code"] == "internal_server_error"
    assert body["error"]["message"] == "Unexpected internal server error."
    assert "database password" not in body["error"]["message"]
    assert body["error"]["details"] == {}


def test_verify_video_endpoint_rejects_invalid_expected_actions():
    app.dependency_overrides[get_video_verification_orchestrator_service] = (
        lambda: FakeInvalidActionVideoVerificationOrchestratorService()
    )

    client = TestClient(app)

    response = client.post(
        "/api/verify-video",
        json={
            "correlation_id": "trace-invalid-actions-1",
            "challenge_id": "backend-challenge-123",
            "expected_actions": [
                "center",
                "right",
                "left",
                "up",
                "down",
            ],
            "national_id_image_uri": "/tmp/id.jpg",
            "video_uri": "/tmp/video.mp4",
            "face_match_level": 2,
            "liveness_level": 2,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 400

    body = response.json()

    assert body["ok"] is False
    assert body["correlation_id"] == "trace-invalid-actions-1"
    assert body["processing_time_ms"] >= 0
    assert body["error"]["code"] == "invalid_request"
    assert "Invalid expected_actions" in body["error"]["message"]
