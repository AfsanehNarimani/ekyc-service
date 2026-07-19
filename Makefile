.PHONY: run test test-fast lint format check clean

run:
	uvicorn app:app --reload

test:
	pytest

test-fast:
	pytest tests

lint:
	ruff check api core face liveness media services storage video tests app.py config.py

format:
	black api core face liveness media services storage video tests app.py config.py

check:
	ruff check api core face liveness media services storage video tests app.py config.py
	pytest

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name ".pytest_cache" -type d -exec rm -rf {} +
	find . -name ".ruff_cache" -type d -exec rm -rf {} +
