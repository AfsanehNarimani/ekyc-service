.PHONY: install install-dev run test test-fast lint format check clean

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

run:
	uvicorn app:app --reload

test:
	pytest

test-fast:
	pytest tests

lint:
	ruff check api authentication core face liveness media services storage video tests app.py config.py

format:
	black api authentication core face liveness media services storage video tests app.py config.py

check:
	ruff check api authentication core face liveness media services storage video tests app.py config.py
	pytest

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name ".pytest_cache" -type d -exec rm -rf {} +
	find . -name ".ruff_cache" -type d -exec rm -rf {} +
