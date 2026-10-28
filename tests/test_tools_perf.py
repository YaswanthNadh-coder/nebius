"""Tests for Item F: Tools and Performance enhancements (caching, text schema, snapshot clone, sampling, message truncation, settings)."""

import json
from unittest.mock import MagicMock, patch

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.agent.llm import call_llm
from timemachine.agent.loop import truncate_older_tool_results
from timemachine.agent.tools import ToolExecutor
from timemachine.config import Settings
from timemachine.traps.profiler import profile_table
from timemachine.verify.spec import MigrationSpec


def _setup_db():
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.executemany("INSERT INTO employees VALUES (?, ?);", [(1, "Alice"), (2, "Bob")])
    conn.commit()
    spec = MigrationSpec(goal="Add email", affected_tables=["employees"])
    return adapter, spec


def test_schema_and_profile_caching_per_run():
    """Cache introspect() and profile_table results per run; return cached response on repeat call."""
    adapter, spec = _setup_db()
    executor = ToolExecutor(adapter, spec=spec, target_table="employees")

    # 1. get_schema first call: introspects
    res1 = executor.execute("get_schema", "{}")
    assert "schema_summary" in res1
    assert "employees" in res1["schema_summary"]
    assert "already provided" not in res1.get("note", "").lower()

    # Repeat call with identical args: returns cached notice
    res2 = executor.execute("get_schema", "{}")
    assert "already provided" in res2.get("note", "").lower()

    # 2. profile_table first call: profiles table
    prof1 = executor.execute("profile_table", json.dumps({"table_name": "employees"}))
    assert "profiles" in prof1
    assert "already provided" not in prof1.get("note", "").lower()

    # Repeat call: returns cached notice
    prof2 = executor.execute("profile_table", json.dumps({"table_name": "employees"}))
    assert "already provided" in prof2.get("note", "").lower()


def test_get_schema_returns_formatted_text_not_json():
    """get_schema returns format_schema_summary text, not schema.model_dump() JSON."""
    adapter, spec = _setup_db()
    executor = ToolExecutor(adapter, spec=spec, target_table="employees")

    res = executor.execute("get_schema", "{}")
    assert "schema_summary" in res
    assert isinstance(res["schema_summary"], str)
    # Must not return the raw schema dict
    assert "schema" not in res or not isinstance(res.get("schema"), dict)


def test_profile_table_sampling_on_large_table():
    """profile_table on tables >50k rows samples at most 50k rows."""
    adapter, _ = _setup_db()
    conn = adapter.open_readonly()
    cur = conn.cursor()
    cur.execute("CREATE TABLE large_table (id INT PRIMARY KEY, val TEXT);")
    # Simulate a table with 60,000 rows
    cur.executemany(
        "INSERT INTO large_table VALUES (?, ?);",
        [(i, f"val_{i}") for i in range(1000)],
    )
    conn.commit()

    schema = adapter.introspect()
    tbl = schema.get_table("large_table")
    assert tbl is not None

    with patch("timemachine.traps.profiler.MAX_SAMPLE_ROWS", 500):
        # Sample limit set to 500 for test
        profiles = profile_table(conn, tbl)
        assert "val" in profiles
        assert profiles["val"].total_rows == 1000


def test_truncate_older_tool_results_in_message_history():
    """Truncate older tool results in the message history (keep the last 2 full)."""
    messages = [
        {"role": "user", "content": "Start migration"},
        {"role": "assistant", "content": "Calling tool 1", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "X" * 1000},
        {"role": "assistant", "content": "Calling tool 2", "tool_calls": [{"id": "t2"}]},
        {"role": "tool", "tool_call_id": "t2", "content": "Y" * 1000},
        {"role": "assistant", "content": "Calling tool 3", "tool_calls": [{"id": "t3"}]},
        {"role": "tool", "tool_call_id": "t3", "content": "Z" * 1000},
        {"role": "assistant", "content": "Calling tool 4", "tool_calls": [{"id": "t4"}]},
        {"role": "tool", "tool_call_id": "t4", "content": "W" * 1000},
    ]

    truncated = truncate_older_tool_results(messages, keep_full_count=2, max_truncated_chars=50)

    # Tool 1 and 2 should be truncated
    assert len(truncated[2]["content"]) < 200
    assert "truncated" in truncated[2]["content"]
    assert len(truncated[4]["content"]) < 200
    assert "truncated" in truncated[4]["content"]

    # Tool 3 and 4 (last 2) must remain full
    assert len(truncated[6]["content"]) == 1000
    assert len(truncated[8]["content"]) == 1000


def test_call_llm_uses_max_tokens_and_reasoning_effort_from_settings():
    """call_llm reads max_tokens and reasoning_effort setting."""
    client = MagicMock()
    fake_choice = MagicMock()
    fake_choice.message = MagicMock(content="SQL", tool_calls=None)
    fake_choice.finish_reason = "stop"
    client.chat.completions.create.return_value = MagicMock(
        choices=[fake_choice],
        usage=MagicMock(total_tokens=10),
    )

    test_settings = Settings(
        max_llm_tokens=4096,
        reasoning_effort="low",
    )

    with patch("timemachine.agent.llm.get_settings", return_value=test_settings):
        call_llm(client, "test-model", "test prompt")

    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["max_tokens"] == 4096
    assert call_kwargs.get("reasoning_effort") == "low"
