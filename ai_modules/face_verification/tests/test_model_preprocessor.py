import numpy as np
from PIL import Image

from core.face_model_profile import FaceModelProfile
from face.model_preprocessor import ModelPreprocessor


def test_model_preprocessor_builds_nchw_tensor():
    profile = FaceModelProfile(
        name="edgeface",
        input_size=(112, 112),
        input_layout="NCHW",
        color_order="RGB",
        normalization="minus_127_5_div_128",
    )

    tensor = ModelPreprocessor(profile).preprocess(
        Image.new("RGB", (300, 400), color=(128, 128, 128))
    )

    assert tensor.shape == (1, 3, 112, 112)
    assert tensor.dtype == np.float32


def test_model_preprocessor_builds_nhwc_tensor():
    profile = FaceModelProfile(
        name="ghostfacenet",
        input_size=(112, 112),
        input_layout="NHWC",
        color_order="RGB",
        normalization="minus_127_5_div_128",
    )

    tensor = ModelPreprocessor(profile).preprocess(
        Image.new("RGB", (300, 400), color=(128, 128, 128))
    )

    assert tensor.shape == (1, 112, 112, 3)
    assert tensor.dtype == np.float32


def test_model_preprocessor_builds_facenet_160_tensor():
    profile = FaceModelProfile(
        name="facenet",
        input_size=(160, 160),
        input_layout="NCHW",
        color_order="RGB",
        normalization="minus_127_5_div_128",
    )

    tensor = ModelPreprocessor(profile).preprocess(
        Image.new("RGB", (300, 400), color=(128, 128, 128))
    )

    assert tensor.shape == (1, 3, 160, 160)
    assert tensor.dtype == np.float32


def test_model_preprocessor_builds_adaface_bgr_tensor():
    profile = FaceModelProfile(
        name="adaface",
        input_size=(112, 112),
        input_layout="NCHW",
        color_order="BGR",
        normalization="zero_to_one_minus_0_5_div_0_5",
    )

    tensor = ModelPreprocessor(profile).preprocess(
        Image.new("RGB", (300, 400), color=(128, 128, 128))
    )

    assert tensor.shape == (1, 3, 112, 112)
    assert tensor.dtype == np.float32
