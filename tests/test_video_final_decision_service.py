from services.video_final_decision_service import VideoFinalDecisionService


def test_video_final_decision_service_approves_when_all_enforced_checks_pass():
    decision = VideoFinalDecisionService().decide(
        face_match_passed=True,
        actions_passed=True,
        liveness_passed=True,
        passive_liveness_passed=True,
        passive_liveness_enforced=True,
    )

    assert decision.approved is True
    assert decision.status == "approved"
    assert decision.rejection_reasons == []
    assert decision.passive_liveness_passed is True
    assert decision.passive_liveness_enforced is True


def test_video_final_decision_service_rejects_passive_liveness_when_enforced():
    decision = VideoFinalDecisionService().decide(
        face_match_passed=True,
        actions_passed=True,
        liveness_passed=True,
        passive_liveness_passed=False,
        passive_liveness_enforced=True,
    )

    assert decision.approved is False
    assert decision.status == "rejected"
    assert decision.rejection_reasons == [
        "passive_liveness_failed",
    ]
    assert decision.passive_liveness_passed is False
    assert decision.passive_liveness_enforced is True


def test_video_final_decision_service_does_not_reject_passive_liveness_in_report_only_mode():
    decision = VideoFinalDecisionService().decide(
        face_match_passed=True,
        actions_passed=True,
        liveness_passed=True,
        passive_liveness_passed=False,
        passive_liveness_enforced=False,
    )

    assert decision.approved is True
    assert decision.status == "approved"
    assert decision.rejection_reasons == []
    assert decision.passive_liveness_passed is False
    assert decision.passive_liveness_enforced is False


def test_video_final_decision_service_rejects_other_failed_checks_even_if_passive_is_report_only():
    decision = VideoFinalDecisionService().decide(
        face_match_passed=False,
        actions_passed=True,
        liveness_passed=False,
        passive_liveness_passed=False,
        passive_liveness_enforced=False,
    )

    assert decision.approved is False
    assert decision.status == "rejected"
    assert decision.rejection_reasons == [
        "face_match_failed",
        "liveness_failed",
    ]
