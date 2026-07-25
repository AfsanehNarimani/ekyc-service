import argparse
import json
from pathlib import Path

from app import app


def export_openapi(
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            app.openapi(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/ekyc-openapi-v0.1.0.json"),
    )

    args = parser.parse_args()

    export_openapi(
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
