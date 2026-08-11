from api.video_verification_response_mapper import VideoVerificationResponseMapper
from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_challenge_check_result import HeadPoseChallengeCheckResult
from core.final_decision import FinalDecision
from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_action_type import HeadPoseActionType
from core.passive_liveness_result import PassiveLivenessResult
from ai_modules.face_verification.domain.video_face_match_result import (
    VideoFaceMatchResult,
)
from core.video_liveness_result import VideoLivenessResult
from core.video_verification_result import VideoVerificationResult


def test_video_verification_response_mapper_maps_result_to_response():
    result = VideoVerificationResult(
        correlation_id="trace-1",
        challenge_id="challenge-1",
        expected_actions=[
            HeadPoseActionType.CENTER,
            HeadPoseActionType.LEFT,
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
        action_check=HeadPoseChallengeCheckResult(
            passed=True,
            required_actions=[
                HeadPoseActionType.CENTER,
                HeadPoseActionType.LEFT,
            ],
            observed_actions=[
                HeadPoseActionType.CENTER,
                HeadPoseActionType.LEFT,
            ],
            compact_actions=[
                HeadPoseActionType.CENTER,
                HeadPoseActionType.LEFT,
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

    response = VideoVerificationResponseMapper().to_response(result)

    assert response.ok is True
    assert response.correlation_id == "trace-1"
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
    assert response.passive_liveness.passed is True
    assert response.passive_liveness.model_name == "stub_passive_liveness"
    assert response.final_decision.approved is True
    assert response.final_decision.passive_liveness_passed is True
    assert response.processing_time_ms == 123.45
    assert response.final_decision.passive_liveness_enforced is False
