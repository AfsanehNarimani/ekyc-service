from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FinalDecision:
    approved: bool
    status: str
    rejection_reasons: list[str]
    face_match_passed: bool
    actions_passed: bool
    liveness_passed: bool
    passive_liveness_passed: bool
    passive_liveness_enforced: bool
