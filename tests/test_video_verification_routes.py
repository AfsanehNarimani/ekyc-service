from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from api.dependencies import (
    get_image_file_loader,
    get_media_loader,
    get_video_action_check_service,
    get_video_face_match_service,
    get_video_liveness_service,
)
from app import app
from core.liveness_action import LivenessAction


class FakeMediaLoader:
    def load_to_file(
        self,
        reference,
        suffix: str,
    ) -> Path:
        return Path(f"/tmp/fake{suffix}")


class FakeImageFileLoader:
    def load(
        self,
        path: Path,
    ) -> Image.Image:
        return Image.new("RGB", (112, 112))


class FakeVideoFaceMatchService:
    def match_id_card_to_video(
        self,
        national_id_image,
        video_path: Path,
        strictness_level: int,
    ):
        return SimpleNamespace(
            is_match=True,
            score=0.76,
            best_score=0.82,
            threshold=0.55,
            model_name="edgeface",
            processed_frames=10,
            skipped_frames=2,
        )


class FakeVideoActionCheckService:
    def check_actions(
        self,
        video_path: Path,
        expected_actions: list[LivenessAction],
    ):
        return SimpleNamespace(
            passed=True,
            required_actions=expected_actions,
            observed_actions=[
                LivenessAction.CENTER,
                LivenessAction.LEFT,
                LivenessAction.RIGHT,
                LivenessAction.UP,
                LivenessAction.DOWN,
            ],
            compact_actions=[
                LivenessAction.CENTER,
                LivenessAction.LEFT,
                LivenessAction.RIGHT,
                LivenessAction.UP,
                LivenessAction.DOWN,
            ],
            missing_actions=[],
            processed_frames=10,
            skipped_frames=0,
        )


class FakeVideoLivenessService:
    def check_liveness(
        self,
        video_path: Path,
        strictness_level: int,
    ):
        return SimpleNamespace(
            passed=True,
            strictness_level=strictness_level,
            processed_frames=10,
            skipped_frames=0,
            min_required_frames=5,
            pose_variation_score=30.0,
            min_pose_variation_score=18.0,
            reason="passed_basic_liveness",
        )


def test_verify_video_endpoint_returns_final_approved_result():
    app.dependency_overrides[get_media_loader] = lambda: FakeMediaLoader()
    app.dependency_overrides[get_image_file_loader] = lambda: FakeImageFileLoader()
    app.dependency_overrides[get_video_face_match_service] = (
        lambda: FakeVideoFaceMatchService()
    )
    app.dependency_overrides[get_video_action_check_service] = (
        lambda: FakeVideoActionCheckService()
    )
    app.dependency_overrides[get_video_liveness_service] = (
        lambda: FakeVideoLivenessService()
    )

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
    assert body["challenge_id"] == "backend-challenge-123"

    assert body["face_match"]["passed"] is True
    assert body["action_check"]["passed"] is True
    assert body["liveness"]["passed"] is True

    assert body["final_decision"]["approved"] is True
    assert body["final_decision"]["status"] == "approved"
    assert body["final_decision"]["rejection_reasons"] == []
    assert body["final_decision"]["face_match_passed"] is True
    assert body["final_decision"]["actions_passed"] is True
    assert body["final_decision"]["liveness_passed"] is True
