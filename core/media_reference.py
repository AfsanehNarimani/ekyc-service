from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MediaReference:
    uri: str

    def is_s3_uri(self) -> bool:
        return self.uri.startswith("s3://")

    def is_http_uri(self) -> bool:
        return self.uri.startswith("http://") or self.uri.startswith("https://")

    def is_local_path(self) -> bool:
        return not self.is_s3_uri() and not self.is_http_uri()
