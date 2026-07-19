from services.video_final_decision_service import VideoFinalDecisionService


def test_video_final_decision_service_approves_when_all_checks_pass():
    decision = VideoFinalDecisionService().decide(
        face_match_passed=True,
        actions_passed=True,
        liveness_passed=True,
    )

    assert decision.approved is True
    assert decision.status == "approved"
    assert decision.rejection_reasons == []


def test_video_final_decision_service_rejects_failed_checks():
    decision = VideoFinalDecisionService().decide(
        face_match_passed=False,
        actions_passed=True,
        liveness_passed=False,
    )

    assert decision.approved is False
    assert decision.status == "rejected"
    assert decision.rejection_reasons == [
        "face_match_failed",
        "liveness_failed",
    ]
