from api.video_verification_schemas import VerifyVideoResponse
from core.video_verification_result import VideoVerificationResult


class VideoVerificationResponseMapper:
    def to_response(
        self,
        result: VideoVerificationResult,
    ) -> VerifyVideoResponse:
        return VerifyVideoResponse(
            ok=True,
            trace_id=result.trace_id,
            challenge_id=result.challenge_id,
            expected_actions=[
                action.value for action in result.expected_actions
            ],
            face_match={
                "passed": result.face_match.is_match,
                "score": result.face_match.score,
                "best_score": result.face_match.best_score,
                "threshold": result.face_match.threshold,
                "model_name": result.face_match.model_name,
                "processed_frames": result.face_match.processed_frames,
                "skipped_frames": result.face_match.skipped_frames,
            },
            liveness={
                "passed": result.liveness.passed,
                "strictness_level": result.liveness.strictness_level,
                "processed_frames": result.liveness.processed_frames,
                "skipped_frames": result.liveness.skipped_frames,
                "min_required_frames": result.liveness.min_required_frames,
                "pose_variation_score": result.liveness.pose_variation_score,
                "min_pose_variation_score": (
                    result.liveness.min_pose_variation_score
                ),
                "reason": result.liveness.reason,
            },
            passive_liveness={
                "passed": result.passive_liveness.passed,
                "score": result.passive_liveness.score,
                "threshold": result.passive_liveness.threshold,
                "model_name": result.passive_liveness.model_name,
                "processed_frames": result.passive_liveness.processed_frames,
                "skipped_frames": result.passive_liveness.skipped_frames,
                "reason": result.passive_liveness.reason,
            },
            action_check={
                "passed": result.action_check.passed,
                "required_actions": [
                    action.value for action in result.action_check.required_actions
                ],
                "observed_actions": [
                    action.value for action in result.action_check.observed_actions
                ],
                "compact_actions": [
                    action.value for action in result.action_check.compact_actions
                ],
                "missing_actions": [
                    action.value for action in result.action_check.missing_actions
                ],
                "processed_frames": result.action_check.processed_frames,
                "skipped_frames": result.action_check.skipped_frames,
            },
            final_decision={
                "approved": result.final_decision.approved,
                "status": result.final_decision.status,
                "rejection_reasons": result.final_decision.rejection_reasons,
                "face_match_passed": result.final_decision.face_match_passed,
                "actions_passed": result.final_decision.actions_passed,
                "liveness_passed": result.final_decision.liveness_passed,
                "passive_liveness_passed": (
                    result.final_decision.passive_liveness_passed
                ),
            },
            processing_time_ms=result.processing_time_ms,
        )
