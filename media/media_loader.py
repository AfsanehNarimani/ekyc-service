from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import boto3
import requests

from core.exceptions import MediaLoadError
from core.media_reference import MediaReference


class MediaLoader:
    def __init__(
        self,
        temp_dir: Path,
        s3_region_name: str | None = None,
    ) -> None:
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.s3_client = boto3.client(
            "s3",
            region_name=s3_region_name,
        )

    def load_to_file(
        self,
        reference: MediaReference,
        suffix: str,
    ) -> Path:
        if reference.is_s3_uri():
            return self._download_s3_uri(
                uri=reference.uri,
                suffix=suffix,
            )

        if reference.is_http_uri():
            return self._download_http_uri(
                uri=reference.uri,
                suffix=suffix,
            )

        return self._load_local_file(
            path=Path(reference.uri),
        )

    def _download_s3_uri(
        self,
        uri: str,
        suffix: str,
    ) -> Path:
        parsed = urlparse(uri)

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        if not bucket or not key:
            raise MediaLoadError(f"Invalid S3 URI: {uri}")

        output_path = self._make_temp_path(suffix=suffix)

        try:
            self.s3_client.download_file(
                Bucket=bucket,
                Key=key,
                Filename=str(output_path),
            )
        except Exception as error:
            raise MediaLoadError(f"Could not download S3 object: {uri}") from error

        return output_path

    def _download_http_uri(
        self,
        uri: str,
        suffix: str,
    ) -> Path:
        output_path = self._make_temp_path(suffix=suffix)

        try:
            response = requests.get(
                uri,
                timeout=60,
            )
            response.raise_for_status()
            output_path.write_bytes(response.content)
        except Exception as error:
            raise MediaLoadError(f"Could not download HTTP media: {uri}") from error

        return output_path

    def _load_local_file(
        self,
        path: Path,
    ) -> Path:
        if not path.exists():
            raise FileNotFoundError(path)

        return path

    def _make_temp_path(
        self,
        suffix: str,
    ) -> Path:
        if not suffix.startswith("."):
            suffix = f".{suffix}"

        return self.temp_dir / f"{uuid4().hex}{suffix}"
