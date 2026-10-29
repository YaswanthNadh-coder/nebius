"""Tests for Item G: Optimizations (single-query profiling, caching, packaging)."""

from pathlib import Path

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.traps.profiler import (
    clear_profiles_cache,
    get_cached_profile,
    profile_database,
    set_cached_profile,
)


def test_profile_database_caching():
    """Profiles are cached per (fingerprint, inject_traps) and reused."""
    clear_profiles_cache()
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.clone()
    schema = adapter.introspect()

    assert get_cached_profile("test_fp", True) is None

    p1 = profile_database(conn, schema)
    set_cached_profile("test_fp", True, p1)

    cached = get_cached_profile("test_fp", True)
    assert cached is not None
    assert cached == p1
    conn.close()


def test_requirements_split():
    """requirements.txt and requirements-dev.txt are properly separated."""
    root = Path(__file__).resolve().parent.parent
    req_runtime = (root / "requirements.txt").read_text()
    req_dev = (root / "requirements-dev.txt").read_text()

    assert "pytest" in req_dev
    assert "ruff" in req_dev
    assert "pyflakes" in req_dev
    assert "streamlit" in req_runtime
    assert "pytest" not in req_runtime


def test_dockerfile_healthcheck_urllib():
    """Dockerfile uses python urllib instead of curl for healthcheck."""
    root = Path(__file__).resolve().parent.parent
    dockerfile = (root / "Dockerfile").read_text()
    assert "urllib" in dockerfile
    assert "curl" not in dockerfile
