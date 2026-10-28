"""Tests for live smoke script helpers, attempts table formatting, and performance benchmarks."""

import io

from scripts.live_smoke import benchmark_performance, print_attempts_table
from timemachine.agent.loop import AttemptRecord


def test_print_attempts_table_output():
    """Attempts table prints formatted rows with required columns."""
    attempts = [
        AttemptRecord(
            attempt=1,
            model="meta/llama-3.3-70b-instruct",
            passed_count=8,
            failed_count=2,
            first_failed_check="DATA_MISMATCH: last_name 1 wrong",
            sandbox_error=None,
            latency_ms=1250.5,
            tokens=450,
            tool_rounds_used=2,
            no_sql_replies=0,
        ),
        AttemptRecord(
            attempt=2,
            model="nvidia/llama-3.1-nemotron-70b-reward",
            passed_count=10,
            failed_count=0,
            first_failed_check="",
            sandbox_error=None,
            latency_ms=980.2,
            tokens=320,
            tool_rounds_used=1,
            no_sql_replies=0,
        ),
    ]

    buf = io.StringIO()
    import sys
    orig_stdout = sys.stdout
    try:
        sys.stdout = buf
        print_attempts_table(attempts)
    finally:
        sys.stdout = orig_stdout

    out = buf.getvalue()
    assert "Attempt" in out
    assert "Model" in out
    assert "Passed" in out
    assert "Failed" in out
    assert "First Failed Check" in out
    assert "Sandbox Error" in out
    assert "Latency ms" in out
    assert "Tokens" in out
    assert "DATA_MISMATCH" in out


def test_benchmark_performance_execution():
    """Benchmark runs at synthetic row counts and returns valid timings."""
    # Run with small counts to test quickly in unit test
    results = benchmark_performance(row_counts=[1000])
    assert "1000_rows" in results
    r = results["1000_rows"]
    assert "clone_ms" in r
    assert "profile_ms" in r
    assert "run_on_clone_ms" in r
    assert r["clone_ms"] >= 0
    assert r["profile_ms"] >= 0
    assert r["run_on_clone_ms"] >= 0
