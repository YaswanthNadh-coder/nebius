"""Tests for live smoke script helpers, attempts table formatting, and performance benchmarks."""

import io

from scripts.live_smoke import (
    assert_no_up_contains_down,
    benchmark_performance,
    print_attempts_table,
    serialize_attempts,
)
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


def test_serialize_attempts_records_all_fields():
    """serialize_attempts records up_sql, down_sql, splitter rule, model, failed checks, tool rounds, tokens."""
    attempts = [
        AttemptRecord(
            attempt=1,
            model="nvidia/nemotron-3-super-120b-a12b",
            passed_count=5,
            failed_count=1,
            first_failed_check="NULL_CORRUPTION: Column salary",
            sandbox_error=None,
            latency_ms=850.0,
            tokens=520,
            tool_rounds_used=2,
            no_sql_replies=0,
            up_sql="CREATE TABLE employees_new (id INT);",
            down_sql="CREATE TABLE employees_old (id INT);",
            split_rule="rule_fence_tag_down",
        )
    ]

    serialized = serialize_attempts(attempts)
    assert len(serialized) == 1
    rec = serialized[0]
    assert rec["up_sql_preview"].startswith("CREATE TABLE employees_new")
    assert rec["down_sql_preview"].startswith("CREATE TABLE employees_old")
    assert rec["split_rule"] == "rule_fence_tag_down"
    assert rec["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert "NULL_CORRUPTION" in rec["first_failed_check"]
    assert rec["tool_rounds_used"] == 2
    assert rec["tokens"] == 520


def test_assert_no_up_contains_down_fails_when_contained():
    """assert_no_up_contains_down raises AssertionError if UP contains DOWN script."""
    down_code = "DROP TABLE employees;\nALTER TABLE employees_old RENAME TO employees;"
    bad_up = f"CREATE TABLE employees_new (id INT);\n-- DOWN SCRIPT LEAKED\n{down_code}"

    attempts = [
        AttemptRecord(
            attempt=1,
            model="model",
            up_sql=bad_up,
            down_sql=down_code,
            split_rule="none",
        )
    ]

    import pytest
    with pytest.raises(AssertionError) as exc_info:
        assert_no_up_contains_down(attempts, run_label="test_run")
    assert "contains down_sql" in str(exc_info.value).lower() or "down marker" in str(exc_info.value).lower()

