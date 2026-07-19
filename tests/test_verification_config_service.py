from services.verification_config_service import VerificationConfigService


def test_verification_config_service_returns_supported_levels():
    config = VerificationConfigService().get_config()

    assert config["default_face_model"] == "edgeface"

    assert config["supported_actions"] == [
        "center",
        "left",
        "right",
        "up",
        "down",
    ]

    assert config["face_match_levels"]["1"]["name"] == "relaxed"
    assert config["face_match_levels"]["2"]["name"] == "normal"
    assert config["face_match_levels"]["3"]["name"] == "strict"

    assert config["liveness_levels"]["1"]["name"] == "relaxed"
    assert config["liveness_levels"]["2"]["name"] == "normal"
    assert config["liveness_levels"]["3"]["name"] == "strict"

    assert config["passive_liveness_levels"]["1"]["name"] == "relaxed"
    assert config["passive_liveness_levels"]["2"]["name"] == "normal"
    assert config["passive_liveness_levels"]["3"]["name"] == "strict"
    assert config["passive_liveness_levels"]["2"]["threshold"] == 0.6
