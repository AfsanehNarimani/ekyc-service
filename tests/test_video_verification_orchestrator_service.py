from pathlib import Path

from PIL import Image

from core.action_check_result import ActionCheckResult
from core.final_decision import FinalDecision
from core.liveness_action import LivenessAction
from core.passive_liveness_result import PassiveLivenessResult
from ai_modules.face_verification.domain.video_face_match_result import VideoFaceMatchResult
from core.video_liveness_result import VideoLivenessResult
from liveness.expected_action_policy import ExpectedActionPolicy
from services.video_verification_orchestrator_service import (
    VideoVerificationOrchestratorService,
)


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
        video_analysis,
        strictness_level: int,
    ) -> VideoFaceMatchResult:
        return VideoFaceMatchResult(
            is_match=True,
            score=0.76,
            best_score=0.82,
            threshold=0.55,
            model_name="edgeface",
            processed_frames=10,
            skipped_frames=0,
            frame_scores=[0.76],
        )


class FakeVideoActionCheckService:
    def check_actions(
        self,
        video_analysis,
        expected_actions: list[LivenessAction],
    ) -> ActionCheckResult:
        return ActionCheckResult(
            passed=True,
            required_actions=expected_actions,
            observed_actions=expected_actions,
            compact_actions=expected_actions,
            missing_actions=[],
            processed_frames=10,
            skipped_frames=0,
        )


class FakeVideoLivenessService:
    def check_liveness(
        self,
        video_analysis,
        strictness_level: int,
    ) -> VideoLivenessResult:
        return VideoLivenessResult(
            passed=True,
            strictness_level=strictness_level,
            processed_frames=10,
            skipped_frames=0,
            min_required_frames=5,
            pose_variation_score=30.0,
            min_pose_variation_score=18.0,
            reason="passed_basic_liveness",
        )


class FakePassiveLivenessChecker:
    def check(
        self,
        video_analysis,
        strictness_level: int,
    ) -> PassiveLivenessResult:
        return PassiveLivenessResult(
            passed=True,
            score=1.0,
            threshold=0.6,
            model_name="stub_passive_liveness",
            processed_frames=0,
            skipped_frames=0,
            reason="stub_passive_liveness_passed",
        )


class FakeFinalDecisionService:
    def decide(
        self,
        face_match_passed: bool,
        actions_passed: bool,
        liveness_passed: bool,
        passive_liveness_passed: bool,
        passive_liveness_enforced: bool,
    ) -> FinalDecision:
        approved = (
            face_match_passed
            and actions_passed
            and liveness_passed
            and (passive_liveness_passed or not passive_liveness_enforced)
        )

        return FinalDecision(
            approved=approved,
            status="approved" if approved else "rejected",
            rejection_reasons=[] if approved else ["failed"],
            face_match_passed=face_match_passed,
            actions_passed=actions_passed,
            liveness_passed=liveness_passed,
            passive_liveness_passed=passive_liveness_passed,
            passive_liveness_enforced=passive_liveness_enforced,
        )


class FakeMediaCleanupService:
    def __init__(self) -> None:
        self.cleaned_paths = []

    def cleanup_temp_files(
        self,
        paths,
        correlation_id: str,
    ) -> None:
        self.cleaned_paths.extend(paths)


def make_expected_action_policy() -> ExpectedActionPolicy:
    return ExpectedActionPolicy(
        required_actions=[
            LivenessAction.CENTER,
            LivenessAction.LEFT,
            LivenessAction.RIGHT,
            LivenessAction.UP,
            LivenessAction.DOWN,
        ]
    )


def test_video_verification_orchestrator_runs_full_flow():
    media_cleanup_service = FakeMediaCleanupService()

    video_frame_analysis_service = FakeVideoFrameAnalysisService()

    service = VideoVerificationOrchestratorService(
        media_loader=FakeMediaLoader(),
        image_file_loader=FakeImageFileLoader(),
        video_frame_analysis_service=(video_frame_analysis_service),
        face_match_service=FakeVideoFaceMatchService(),
        action_check_service=FakeVideoActionCheckService(),
        liveness_service=FakeVideoLivenessService(),
        passive_liveness_checker=FakePassiveLivenessChecker(),
        final_decision_service=FakeFinalDecisionService(),
        expected_action_policy=make_expected_action_policy(),
        media_cleanup_service=media_cleanup_service,
    )

    result = service.verify(
        correlation_id="trace-1",
        challenge_id="challenge-1",
        expected_actions=[
            LivenessAction.CENTER,
            LivenessAction.LEFT,
            LivenessAction.RIGHT,
            LivenessAction.UP,
            LivenessAction.DOWN,
        ],
        national_id_image_uri="s3://bucket/id.jpg",
        video_uri="s3://bucket/video.mp4",
        face_match_level=2,
        liveness_level=2,
    )

    assert result.correlation_id == "trace-1"
    assert result.challenge_id == "challenge-1"
    assert result.face_match.is_match is True
    assert result.action_check.passed is True
    assert result.liveness.passed is True
    assert result.passive_liveness.passed is True
    assert result.passive_liveness.model_name == "stub_passive_liveness"
    assert result.final_decision.approved is True
    assert result.final_decision.passive_liveness_passed is True
    assert result.processing_time_ms >= 0

    assert len(media_cleanup_service.cleaned_paths) == 2
    assert media_cleanup_service.cleaned_paths == [
        Path("/tmp/fake.jpg"),
        Path("/tmp/fake.mp4"),
    ]
    assert result.final_decision.passive_liveness_enforced is False

    assert video_frame_analysis_service.call_count == 1


class FakeVideoFrameAnalysisService:
    def __init__(self) -> None:
        self.call_count = 0
        self.analysis = object()

    def analyze(
        self,
        video_path: Path,
    ):
        self.call_count += 1
        return self.analysis
