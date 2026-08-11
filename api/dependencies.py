from functools import lru_cache

# TODO only expose required classes and functions of ai_modules and encapsulate others
from ai_modules.challenge_based_liveness_detection.head_pose_estimation. \
    application.head_pose_challenge_check_service import HeadPoseChallengeCheckService
from ai_modules.challenge_based_liveness_detection.head_pose_estimation \
    .infrastructure.head_pose_action_detector import HeadPoseActionDetector

from ai_modules.challenge_based_liveness_detection.head_pose_estimation.domain.head_pose_action_type import (
    HeadPoseActionType,
)
from ai_modules.face_detection.application.face_detection_service import (
    FaceDetectionService,
)
from ai_modules.face_detection.domain.single_face_validator import SingleFaceValidator
from ai_modules.face_detection.infrastructure.insightface_detector import (
    InsightFaceDetector,
)
from ai_modules.face_verification.application.face_verification_service import (
    FaceVerificationService,
)
from ai_modules.face_verification.domain.face_match_threshold_policy import (
    FaceMatchThresholdPolicy,
)
from ai_modules.face_verification.domain.face_matcher import FaceMatcher
from ai_modules.face_verification.infrastructure.embedders.base_embedder import (
    BaseFaceEmbedder,
)
from ai_modules.face_verification.infrastructure.embedders.face_embedder_factory import (
    FaceEmbedderFactory,
)
from ai_modules.face_verification.infrastructure.preprocessing.face_aligner import (
    FaceAligner,
)
from ai_modules.face_verification.infrastructure.preprocessing.face_preprocessor import (
    FacePreprocessor,
)
from api.error_response_factory import ErrorResponseFactory
from api.video_verification_response_mapper import VideoVerificationResponseMapper
from config import settings
from liveness.expected_action_policy import ExpectedActionPolicy
from liveness.liveness_threshold_policy import (
    LivenessThresholdPolicy,
    LivenessThresholds,
)
from liveness.onnx_passive_liveness_model import ONNXPassiveLivenessModel
from liveness.passive_liveness_ensemble_model import (
    PassiveLivenessEnsembleModel,
)
from liveness.passive_liveness_preprocessor import PassiveLivenessPreprocessor
from liveness.passive_liveness_threshold_policy import (
    PassiveLivenessThresholdPolicy,
)
from media.image_file_loader import ImageFileLoader
from media.media_cleanup_service import MediaCleanupService
from media.media_loader import MediaLoader
from services.head_pose_challenge_orchestrator_service import (
    HeadPoseChallengeOrchestratorService,
)
from services.onnx_passive_liveness_checker import ONNXPassiveLivenessChecker
from services.readiness_service import ReadinessService
from services.stub_passive_liveness_checker import StubPassiveLivenessChecker
from services.verification_config_service import VerificationConfigService
from services.video_final_decision_service import VideoFinalDecisionService
from services.video_liveness_service import VideoLivenessService
from video_processing.video_frame_extractor import VideoFrameExtractor


@lru_cache
def get_face_preprocessor() -> FacePreprocessor:
    return FacePreprocessor(
        detector=get_face_detector(),
        validator=get_single_face_validator(),
        aligner=FaceAligner(),
    )


@lru_cache
def get_face_embedder() -> BaseFaceEmbedder:
    factory = FaceEmbedderFactory.from_settings(
        providers=["CPUExecutionProvider"],
    )

    return factory.create(
        model_name=settings.default_face_model,
    )


@lru_cache
def get_face_matcher() -> FaceMatcher:
    return FaceMatcher(
        threshold=settings.face_match_threshold,
    )


@lru_cache
def get_media_loader() -> MediaLoader:
    return MediaLoader(
        temp_dir=settings.media_temp_dir,
        s3_region_name=settings.s3_region_name,
    )


@lru_cache
def get_video_frame_extractor() -> VideoFrameExtractor:
    return VideoFrameExtractor(
        frame_interval_seconds=settings.video_frame_interval_seconds,
        max_frames=settings.video_max_frames,
    )


@lru_cache
def get_face_match_threshold_policy() -> FaceMatchThresholdPolicy:
    return FaceMatchThresholdPolicy(
        level_to_threshold={
            1: settings.face_match_threshold_level_1,
            2: settings.face_match_threshold_level_2,
            3: settings.face_match_threshold_level_3,
        }
    )


@lru_cache
def get_liveness_threshold_policy() -> LivenessThresholdPolicy:
    return LivenessThresholdPolicy(
        level_to_thresholds={
            1: LivenessThresholds(
                min_required_frames=settings.liveness_min_frames_level_1,
                min_pose_variation_score=settings.liveness_pose_variation_level_1,
            ),
            2: LivenessThresholds(
                min_required_frames=settings.liveness_min_frames_level_2,
                min_pose_variation_score=settings.liveness_pose_variation_level_2,
            ),
            3: LivenessThresholds(
                min_required_frames=settings.liveness_min_frames_level_3,
                min_pose_variation_score=settings.liveness_pose_variation_level_3,
            ),
        }
    )


@lru_cache
def get_video_liveness_service() -> VideoLivenessService:
    return VideoLivenessService(
        threshold_policy=get_liveness_threshold_policy(),
    )


@lru_cache
def get_face_verification_service() -> FaceVerificationService:
    return FaceVerificationService(
        preprocessor=get_face_preprocessor(),
        embedder=get_face_embedder(),
        threshold_policy=get_face_match_threshold_policy(),
        neutral_yaw_threshold=settings.head_pose_neutral_yaw_threshold,
        neutral_pitch_threshold=settings.head_pose_neutral_pitch_threshold,
    )


@lru_cache
def get_image_file_loader() -> ImageFileLoader:
    return ImageFileLoader()


@lru_cache
def get_head_pose_action_detector() -> HeadPoseActionDetector:
    return HeadPoseActionDetector(
        yaw_threshold=settings.head_pose_yaw_threshold,
        pitch_threshold=settings.head_pose_pitch_threshold,
        neutral_yaw_threshold=settings.head_pose_neutral_yaw_threshold,
        neutral_pitch_threshold=settings.head_pose_neutral_pitch_threshold,
    )


@lru_cache
def get_head_pose_challenge_check_service() -> HeadPoseChallengeCheckService:
    return HeadPoseChallengeCheckService(
        head_pose_action_detector=get_head_pose_action_detector(),
    )


@lru_cache
def get_video_final_decision_service() -> VideoFinalDecisionService:
    return VideoFinalDecisionService()


@lru_cache
def get_verification_config_service() -> VerificationConfigService:
    return VerificationConfigService()


# TODO rename function
@lru_cache
def get_video_verification_orchestrator_service() -> (
    HeadPoseChallengeOrchestratorService
):
    return HeadPoseChallengeOrchestratorService(
        media_loader=get_media_loader(),
        image_file_loader=get_image_file_loader(),
        video_frame_extractor=get_video_frame_extractor(),
        face_detection_service=get_face_detection_service(),
        face_verification_service=get_face_verification_service(),
        head_pose_challenge_check_service=get_head_pose_challenge_check_service(),
        liveness_service=get_video_liveness_service(),
        passive_liveness_checker=get_passive_liveness_checker(),
        final_decision_service=get_video_final_decision_service(),
        expected_action_policy=get_expected_action_policy(),
        media_cleanup_service=get_media_cleanup_service(),
    )


@lru_cache
def get_video_verification_response_mapper() -> VideoVerificationResponseMapper:
    return VideoVerificationResponseMapper()


@lru_cache
def get_error_response_factory() -> ErrorResponseFactory:
    return ErrorResponseFactory()


@lru_cache
def get_expected_action_policy() -> ExpectedActionPolicy:
    return ExpectedActionPolicy(
        required_actions=[
            HeadPoseActionType.CENTER,
            HeadPoseActionType.LEFT,
            HeadPoseActionType.RIGHT,
            HeadPoseActionType.UP,
            HeadPoseActionType.DOWN,
        ]
    )


@lru_cache
def get_media_cleanup_service() -> MediaCleanupService:
    return MediaCleanupService()


@lru_cache
def get_readiness_service() -> ReadinessService:
    return ReadinessService()


@lru_cache
def get_passive_liveness_threshold_policy() -> PassiveLivenessThresholdPolicy:
    return PassiveLivenessThresholdPolicy(
        level_to_threshold={
            1: settings.passive_liveness_threshold_level_1,
            2: settings.passive_liveness_threshold_level_2,
            3: settings.passive_liveness_threshold_level_3,
        }
    )


@lru_cache
def get_passive_liveness_checker():
    if settings.passive_liveness_backend == "stub":
        return StubPassiveLivenessChecker(
            threshold_policy=get_passive_liveness_threshold_policy(),
        )

    if settings.passive_liveness_backend == "onnx":
        return get_onnx_passive_liveness_checker()

    raise ValueError(
        f"Unsupported passive_liveness_backend: {settings.passive_liveness_backend}"
    )


@lru_cache
def get_passive_liveness_preprocessor() -> PassiveLivenessPreprocessor:
    return PassiveLivenessPreprocessor(
        input_width=settings.passive_liveness_input_width,
        input_height=settings.passive_liveness_input_height,
        crop_scale=settings.passive_liveness_crop_scale,
        pixel_divisor=settings.passive_liveness_pixel_divisor,
    )


@lru_cache
def get_onnx_passive_liveness_model() -> ONNXPassiveLivenessModel:
    return ONNXPassiveLivenessModel(
        model_path=settings.passive_liveness_model_path,
        preprocessor=get_passive_liveness_preprocessor(),
        real_class_index=settings.passive_liveness_real_class_index,
    )


@lru_cache
def get_passive_liveness_v1se_preprocessor() -> PassiveLivenessPreprocessor:
    return PassiveLivenessPreprocessor(
        input_width=settings.passive_liveness_input_width,
        input_height=settings.passive_liveness_input_height,
        crop_scale=settings.passive_liveness_v1se_crop_scale,
        pixel_divisor=settings.passive_liveness_pixel_divisor,
    )


@lru_cache
def get_onnx_passive_liveness_v1se_model() -> ONNXPassiveLivenessModel:
    return ONNXPassiveLivenessModel(
        model_path=settings.passive_liveness_v1se_model_path,
        preprocessor=get_passive_liveness_v1se_preprocessor(),
        real_class_index=settings.passive_liveness_real_class_index,
    )


@lru_cache
def get_passive_liveness_ensemble_model() -> PassiveLivenessEnsembleModel:
    return PassiveLivenessEnsembleModel(
        models=[
            get_onnx_passive_liveness_model(),
            get_onnx_passive_liveness_v1se_model(),
        ],
        real_class_index=settings.passive_liveness_real_class_index,
    )


@lru_cache
def get_face_detector() -> InsightFaceDetector:
    return InsightFaceDetector(
        det_thresh=settings.detector_threshold,
        model_name=settings.insightface_model_name,
        root_dir=settings.insightface_root_dir,
        providers=["CPUExecutionProvider"],
    )


@lru_cache
def get_single_face_validator() -> SingleFaceValidator:
    return SingleFaceValidator()


@lru_cache
def get_onnx_passive_liveness_checker() -> ONNXPassiveLivenessChecker:
    return ONNXPassiveLivenessChecker(
        model=get_passive_liveness_ensemble_model(),
        threshold_policy=get_passive_liveness_threshold_policy(),
        model_name=settings.passive_liveness_model_name,
        neutral_yaw_threshold=30.0,
        neutral_pitch_threshold=30.0,
    )


@lru_cache
def get_face_detection_service() -> FaceDetectionService:
    return FaceDetectionService(
        detector=get_face_detector(),
        validator=get_single_face_validator(),
    )
