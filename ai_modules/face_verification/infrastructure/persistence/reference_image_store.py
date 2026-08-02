from pathlib import Path

from PIL import Image


class ReferenceImageStore:
    def __init__(
        self,
        root_dir: Path,
    ) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        person_id: str,
        image: Image.Image,
    ) -> Path:
        path = self.root_dir / f"{person_id}.jpg"

        image.save(
            path,
            format="JPEG",
            quality=95,
        )

        return path
