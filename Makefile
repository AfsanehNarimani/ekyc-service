.PHONY: install install-dev run test test-fast lint format check openapi clean

SOURCE_PATHS = \
	api \
	authentication \
	core \
	ai_modules \
	liveness \
	media \
	services \
	video_processing \
	scripts \
	tests \
	app.py \
	config.py

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

run:
	uvicorn app:app --reload

test:
	python -m pytest

test-fast:
	python -m pytest -q tests ai_modules

lint:
	ruff check $(SOURCE_PATHS)

format:
	black $(SOURCE_PATHS)

check:
	ruff check $(SOURCE_PATHS)
	python -m pytest

openapi:
	python -m scripts.export_openapi

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name ".pytest_cache" -type d -exec rm -rf {} +
	find . -name ".ruff_cache" -type d -exec rm -rf {} +