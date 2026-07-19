import re


class PersonIDValidator:
    PATTERN = re.compile(r"^[a-zA-Z0-9_-]{3,64}$")

    def validate(
        self,
        person_id: str,
    ) -> None:
        if not self.PATTERN.fullmatch(person_id):
            raise ValueError(
                "person_id must be 3-64 characters and contain only letters, numbers, '_' or '-'."
            )
