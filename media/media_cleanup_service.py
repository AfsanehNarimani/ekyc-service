import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


class MediaCleanupService:
    def cleanup_temp_file(
        self,
        path: Path,
        correlation_id: str,
    ) -> None:
        try:
            resolved_path = path.resolve()
            temp_root = settings.media_temp_dir.resolve()

            if not self._is_inside_directory(
                path=resolved_path,
                directory=temp_root,
            ):
                return

            if resolved_path.exists() and resolved_path.is_file():
                resolved_path.unlink()
                logger.info(
                    "correlation_id=%s step=temp_file_deleted path=%s",
                    correlation_id,
                    resolved_path.name,
                )

        except Exception:
            logger.exception(
                "correlation_id=%s step=temp_file_cleanup_failed path=%s",
                correlation_id,
                path.name,
            )

    def cleanup_temp_files(
        self,
        paths: list[Path],
        correlation_id: str,
    ) -> None:
        for path in paths:
            self.cleanup_temp_file(
                path=path,
                correlation_id=correlation_id,
            )

    def _is_inside_directory(
        self,
        path: Path,
        directory: Path,
    ) -> bool:
        try:
            path.relative_to(directory)
            return True
        except ValueError:
            return False
