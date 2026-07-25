import pytest

from scripts.calibrate_passive_liveness import (
    build_thresholds,
    read_manifest,
)


def test_read_manifest_loads_calibration_samples(
    tmp_path,
):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")

    manifest_path = tmp_path / "samples.csv"
    manifest_path.write_text(
        (
            "sample_id,label,video_uri,device,"
            "attack_type,notes\n"
            f"real-1,bona_fide,{video_path},"
            "iphone,none,test sample\n"
        ),
        encoding="utf-8",
    )

    samples = read_manifest(manifest_path)

    assert len(samples) == 1
    assert samples[0].sample_id == "real-1"
    assert samples[0].video_path == video_path
    assert samples[0].label.value == "bona_fide"


def test_read_manifest_rejects_missing_columns(
    tmp_path,
):
    manifest_path = tmp_path / "samples.csv"
    manifest_path.write_text(
        "sample_id,label\nreal-1,bona_fide\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Missing manifest columns",
    ):
        read_manifest(manifest_path)


def test_build_thresholds_includes_configured_levels():
    thresholds = build_thresholds(
        step=0.10,
    )

    assert 0.40 in thresholds
    assert 0.60 in thresholds
    assert 0.80 in thresholds


@pytest.mark.parametrize(
    "step",
    [
        0.0,
        -0.1,
        1.1,
    ],
)
def test_build_thresholds_rejects_invalid_step(
    step,
):
    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        build_thresholds(step)
