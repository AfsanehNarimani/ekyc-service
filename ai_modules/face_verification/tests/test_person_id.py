import pytest

from core.person_id import PersonIDValidator


def test_person_id_validator_accepts_valid_id():
    PersonIDValidator().validate("user_123-test")


@pytest.mark.parametrize(
    "person_id",
    [
        "",
        "ab",
        "../user",
        "user/name",
        "user name",
        "a" * 65,
    ],
)
def test_person_id_validator_rejects_invalid_id(person_id):
    with pytest.raises(ValueError):
        PersonIDValidator().validate(person_id)
