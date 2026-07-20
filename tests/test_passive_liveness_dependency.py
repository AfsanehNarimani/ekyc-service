import pytest

import api.dependencies as dependencies
from config import settings
from services.stub_passive_liveness_checker import StubPassiveLivenessChecker


def clear_passive_liveness_dependency_cache():
    dependencies.get_passive_liveness_checker.cache_clear()


def test_passive_liveness_dependency_returns_stub_by_default(monkeypatch):
    monkeypatch.setattr(settings, "passive_liveness_backend", "stub")

    clear_passive_liveness_dependency_cache()

    checker = dependencies.get_passive_liveness_checker()

    assert isinstance(checker, StubPassiveLivenessChecker)


def test_passive_liveness_dependency_rejects_unknown_backend(monkeypatch):
    monkeypatch.setattr(settings, "passive_liveness_backend", "bad-backend")

    clear_passive_liveness_dependency_cache()

    with pytest.raises(ValueError):
        dependencies.get_passive_liveness_checker()

    monkeypatch.setattr(settings, "passive_liveness_backend", "stub")
    clear_passive_liveness_dependency_cache()
