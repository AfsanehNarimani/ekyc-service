from pathlib import Path

from PIL import Image, UnidentifiedImageError

from core.exceptions import MediaLoadError


class ImageFileLoader:
    def load(
        self,
        path: Path,
    ) -> Image.Image:
        try:
            with Image.open(path) as image:
                return image.convert("RGB").copy()
        except UnidentifiedImageError as error:
            raise MediaLoadError(f"Could not load image file: {path}") from error
