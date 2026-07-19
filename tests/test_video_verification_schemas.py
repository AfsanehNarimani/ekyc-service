from api.video_verification_schemas import VerifyVideoResponse


def test_verify_video_response_schema_accepts_valid_response():
    response = VerifyVideoResponse(
        ok=True,
        trace_id="trace-1",
        challenge_id="challenge-1",
        expected_actions=[
            "center",
            "left",
            "right",
            "up",
            "down",
        ],
        face_match={
            "passed": True,
            "score": 0.76,
            "best_score": 0.82,
            "threshold": 0.55,
            "model_name": "edgeface",
            "processed_frames": 10,
            "skipped_frames": 0,
        },
        liveness={
            "passed": True,
            "strictness_level": 2,
            "processed_frames": 10,
            "skipped_frames": 0,
            "min_required_frames": 5,
            "pose_variation_score": 30.0,
            "min_pose_variation_score": 18.0,
            "reason": "passed_basic_liveness",
        },
        action_check={
            "passed": True,
            "required_actions": [
                "center",
                "left",
                "right",
                "up",
                "down",
            ],
            "observed_actions": [
                "center",
                "left",
                "right",
                "up",
                "down",
            ],
            "compact_actions": [
                "center",
                "left",
                "right",
                "up",
                "down",
            ],
            "missing_actions": [],
            "processed_frames": 10,
            "skipped_frames": 0,
        },
        final_decision={
            "approved": True,
            "status": "approved",
            "rejection_reasons": [],
            "face_match_passed": True,
            "actions_passed": True,
            "liveness_passed": True,
        },
        processing_time_ms=123.45,
    )

    assert response.ok is True
    assert response.final_decision.approved is True
    assert response.face_match.model_name == "edgeface"
    assert response.trace_id == "trace-1"
    assert response.processing_time_ms == 123.45
