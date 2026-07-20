from core.final_decision import FinalDecision


class VideoFinalDecisionService:
    def decide(
        self,
        face_match_passed: bool,
        actions_passed: bool,
        liveness_passed: bool,
        passive_liveness_passed: bool,
        passive_liveness_enforced: bool,
    ) -> FinalDecision:
        rejection_reasons: list[str] = []

        if not face_match_passed:
            rejection_reasons.append("face_match_failed")

        if not actions_passed:
            rejection_reasons.append("action_check_failed")

        if not liveness_passed:
            rejection_reasons.append("liveness_failed")

        if passive_liveness_enforced and not passive_liveness_passed:
            rejection_reasons.append("passive_liveness_failed")

        approved = len(rejection_reasons) == 0

        return FinalDecision(
            approved=approved,
            status="approved" if approved else "rejected",
            rejection_reasons=rejection_reasons,
            face_match_passed=face_match_passed,
            actions_passed=actions_passed,
            liveness_passed=liveness_passed,
            passive_liveness_passed=passive_liveness_passed,
            passive_liveness_enforced=passive_liveness_enforced,
        )
