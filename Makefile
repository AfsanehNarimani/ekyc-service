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
    # TODO: How just including tests and ai_modules is a fast test?
	python -m pytest -q tests ai_modules

lint:
	ruff check $(SOURCE_PATHS)

format:
	ruff check --fix $(SOURCE_PATHS)
	black $(SOURCE_PATHS)

check: lint
	python -m pytest

openapi:
	python -m scripts.export_openapi

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name ".pytest_cache" -type d -exec rm -rf {} +
	find . -name ".ruff_cache" -type d -exec rm -rf {} +
