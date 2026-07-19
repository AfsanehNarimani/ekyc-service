from core.action_check_result import ActionCheckResult
from core.final_decision import FinalDecision
from core.liveness_action import LivenessAction
from core.video_face_match_result import VideoFaceMatchResult
from core.video_liveness_result import VideoLivenessResult
from core.video_verification_result import VideoVerificationResult
from api.video_verification_response_mapper import VideoVerificationResponseMapper


def test_video_verification_response_mapper_maps_result_to_response():
    result = VideoVerificationResult(
        trace_id="trace-1",
        challenge_id="challenge-1",
        expected_actions=[
            LivenessAction.CENTER,
            LivenessAction.LEFT,
        ],
        face_match=VideoFaceMatchResult(
            is_match=True,
            score=0.76,
            best_score=0.82,
            threshold=0.55,
            model_name="edgeface",
            processed_frames=10,
            skipped_frames=1,
            frame_scores=[0.76],
        ),
        action_check=ActionCheckResult(
            passed=True,
            required_actions=[
                LivenessAction.CENTER,
                LivenessAction.LEFT,
            ],
            observed_actions=[
                LivenessAction.CENTER,
                LivenessAction.LEFT,
            ],
            compact_actions=[
                LivenessAction.CENTER,
                LivenessAction.LEFT,
            ],
            missing_actions=[],
            processed_frames=10,
            skipped_frames=0,
        ),
        liveness=VideoLivenessResult(
            passed=True,
            strictness_level=2,
            processed_frames=10,
            skipped_frames=0,
            min_required_frames=5,
            pose_variation_score=30.0,
            min_pose_variation_score=18.0,
            reason="passed_basic_liveness",
        ),
        final_decision=FinalDecision(
            approved=True,
            status="approved",
            rejection_reasons=[],
            face_match_passed=True,
            actions_passed=True,
            liveness_passed=True,
        ),
    )

    response = VideoVerificationResponseMapper().to_response(result)

    assert response.ok is True
    assert response.challenge_id == "challenge-1"
    assert response.expected_actions == [
        "center",
        "left",
    ]
    assert response.face_match.passed is True
    assert response.face_match.model_name == "edgeface"
    assert response.action_check.compact_actions == [
        "center",
        "left",
    ]
    assert response.liveness.reason == "passed_basic_liveness"
    assert response.final_decision.approved is True

    assert response.trace_id == "trace-1"
