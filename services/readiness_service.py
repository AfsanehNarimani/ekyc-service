from dataclasses import dataclass
from pathlib import Path

from config import settings


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    passed: bool
    message: str


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    ready: bool
    checks: list[ReadinessCheck]


class ReadinessService:
    def check(self) -> ReadinessResult:
        checks = [
            self._check_default_face_model(),
            self._check_passive_liveness_model(),
            self._check_passive_liveness_decision_mode(),
            self._check_directory(
                name="media_temp_dir",
                path=settings.media_temp_dir,
            ),
            self._check_directory(
                name="reference_dir",
                path=settings.reference_dir,
            ),
            self._check_directory(
                name="embedding_dir",
                path=settings.embedding_dir,
            ),
        ]

        return ReadinessResult(
            ready=all(check.passed for check in checks),
            checks=checks,
        )

    def _check_default_face_model(self) -> ReadinessCheck:
        model_path = self._get_default_model_path()

        if model_path.exists() and model_path.is_file():
            return ReadinessCheck(
                name="default_face_model",
                passed=True,
                message=f"{settings.default_face_model} model file exists",
            )

        return ReadinessCheck(
            name="default_face_model",
            passed=False,
            message=f"{settings.default_face_model} model file is missing",
        )

    def _get_default_model_path(self) -> Path:
        model_path_by_name = {
            "edgeface": settings.edgeface_model_path,
            "ghostfacenet": settings.ghostfacenet_model_path,
            "mobilefacenet": settings.mobilefacenet_model_path,
            "arcface": settings.arcface_model_path,
            "facenet": settings.facenet_model_path,
            "adaface": settings.adaface_model_path,
        }

        model_path = model_path_by_name.get(settings.default_face_model)

        if model_path is None:
            return Path("__missing_model__")

        return model_path

    def _check_directory(
        self,
        name: str,
        path: Path,
    ) -> ReadinessCheck:
        try:
            path.mkdir(
                parents=True,
                exist_ok=True,
            )

            if path.exists() and path.is_dir():
                return ReadinessCheck(
                    name=name,
                    passed=True,
                    message="directory is available",
                )

            return ReadinessCheck(
                name=name,
                passed=False,
                message="path is not a directory",
            )

        except Exception:
            return ReadinessCheck(
                name=name,
                passed=False,
                message="directory is not writable or cannot be created",
            )

    def _check_passive_liveness_model(self) -> ReadinessCheck:
        backend = settings.passive_liveness_backend

        if backend == "stub":
            return ReadinessCheck(
                name="passive_liveness_model",
                passed=True,
                message="model files are not required for stub backend",
            )

        if backend != "onnx":
            return ReadinessCheck(
                name="passive_liveness_model",
                passed=False,
                message=f"unsupported passive liveness backend: {backend}",
            )

        model_paths = {
            "minifasnet_v2": settings.passive_liveness_model_path,
            "minifasnet_v1se": settings.passive_liveness_v1se_model_path,
        }

        missing_models = [
            model_name
            for model_name, model_path in model_paths.items()
            if not model_path.exists() or not model_path.is_file()
        ]

        if not missing_models:
            return ReadinessCheck(
                name="passive_liveness_model",
                passed=True,
                message="all passive liveness model files exist",
            )

        return ReadinessCheck(
            name="passive_liveness_model",
            passed=False,
            message=("missing passive liveness models: " + ", ".join(missing_models)),
        )

    def _check_passive_liveness_decision_mode(self) -> ReadinessCheck:
        decision_mode = settings.passive_liveness_decision_mode
        supported_modes = {
            "report_only",
            "enforce",
        }

        if decision_mode in supported_modes:
            return ReadinessCheck(
                name="passive_liveness_decision_mode",
                passed=True,
                message=f"decision mode is {decision_mode}",
            )

        return ReadinessCheck(
            name="passive_liveness_decision_mode",
            passed=False,
            message=f"unsupported decision mode: {decision_mode}",
        )
