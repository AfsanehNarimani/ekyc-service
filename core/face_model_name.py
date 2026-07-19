ALLOWED_FACE_MODELS = {
    "edgeface",
    "ghostfacenet",
    "mobilefacenet",
    "arcface",
    "facenet",
    "adaface",
}


def validate_face_model_name(
    model_name: str,
) -> None:
    if model_name not in ALLOWED_FACE_MODELS:
        raise ValueError(
            f"Unsupported face model: {model_name}. "
            f"Allowed models: {sorted(ALLOWED_FACE_MODELS)}"
        )
