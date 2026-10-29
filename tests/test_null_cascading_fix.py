"""Tests for cascading NOT NULL elimination, multi-column crash diagnostics, and graceful LLM timeout fallback."""

import sqlite3
from unittest.mock import MagicMock, patch

from openai import APITimeoutError

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.agent.loop import get_model_timeout, run_time_machine_agent
from timemachine.agent.prompts import (
    build_migration_prompt,
    build_repair_prompt,
    get_sqlite_gotchas_block,
)
from timemachine.config import get_settings
from timemachine.verify.invariants import audit_null_crash_diagnostics
from timemachine.verify.spec import ColumnMapping, MigrationSpec


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, content=None, tool_calls=None, finish_reason="stop"):
        self.message = FakeMessage(content, tool_calls)
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, content=None, tool_calls=None, finish_reason="stop"):
        self.choices = [FakeChoice(content, tool_calls, finish_reason)]


def test_audit_null_crash_diagnostics_surfaces_all_nullable_columns():
    """When execution crashes on one NOT NULL column, diagnostic surfaces ALL source columns containing NULLs."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            department TEXT,
            salary INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("INSERT INTO employees (full_name, department, salary) VALUES ('Alice Smith', 'ENG', 90000);")
    # Synthetic trap rows containing NULL in salary, created_at, and department
    conn.execute("INSERT INTO employees (full_name, department, salary, created_at) VALUES ('Trap 1', NULL, NULL, NULL);")
    conn.commit()

    err_msg = "SQLITE EXECUTION CRASH: NOT NULL constraint failed: employees_new.salary"
    current_sql = """
    CREATE TABLE employees_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        department TEXT,
        salary INTEGER NOT NULL,
        created_at TEXT NOT NULL
    );
    INSERT INTO employees_new (id, first_name, last_name, department, salary, created_at)
    SELECT id, 'A', 'B', department, salary, created_at FROM employees;
    """

    diag = audit_null_crash_diagnostics(conn, "employees", err_msg, current_sql=current_sql)
    assert "[MULTI-COLUMN NOT NULL DIAGNOSTIC" in diag
    assert "'salary': 1 NULL row(s)" in diag
    assert "'created_at': 1 NULL row(s)" in diag
    assert "CRITICAL FIX INSTRUCTIONS FOR THIS SINGLE PASS" in diag
    assert "do not add not null to existing columns" in diag.lower()
    assert "wrap its select expression with coalesce" in diag.lower()
    conn.close()


def test_prompt_includes_strict_nullability_preservation_rule():
    """All prompts (gotchas, drafting, repair) include the critical rule against accidental NOT NULL tightening."""
    rule_snippet = (
        "CRITICAL RULE FOR TABLE REBUILDS (CREATE TABLE ..._new): Preserve the exact nullability "
        "of every existing column from the original CREATE TABLE DDL. NEVER add NOT NULL to an "
        "existing nullable column (such as salary, created_at, etc.)"
    )

    gotchas = get_sqlite_gotchas_block()
    assert rule_snippet in gotchas

    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, salary INT);")
    conn.commit()
    schema = adapter.introspect()
    spec = MigrationSpec(goal="Test goal", affected_tables=["employees"])

    draft_p = build_migration_prompt(schema, spec)
    assert rule_snippet in draft_p

    repair_p = build_repair_prompt(schema, spec, "FAILED SQL")
    assert rule_snippet in repair_p


def test_timeout_configuration_and_model_timeouts():
    """Model timeouts are at least 45s for Super and 75s for Ultra, with overall deadline >= 180s."""
    settings = get_settings()
    assert settings.fast_llm_timeout_s >= 45.0
    assert settings.ultra_llm_timeout_s >= 75.0
    assert settings.agent_wall_clock_timeout_s >= 180.0

    assert get_model_timeout("nvidia/nemotron-3-super-120b-a12b", settings) >= 45.0
    assert get_model_timeout("nvidia/Nemotron-3-Ultra-550b-a55b", settings) >= 75.0


def test_ultra_timeout_graceful_fallback_to_super():
    """When Ultra model times out during repair, loop logs fallback event and retries with Super."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT);")
    conn.execute("INSERT INTO employees VALUES (1, 'Alice');")
    conn.commit()

    spec = MigrationSpec(
        goal="Add notes column",
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="id", source_column="id", transform="passthrough"),
                ColumnMapping(new_column="name", source_column="name", transform="passthrough"),
                ColumnMapping(new_column="notes", source_column=None, transform="passthrough", default_value=""),
            ]
        },
    )

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    call_count = 0

    # Attempts 1 and 2: Super returns incomplete SQL causing verification failure
    # Attempt 3: Escalates to Ultra, but Ultra call times out with APITimeoutError!
    # Fallback to Super succeeds with verified SQL.
    def completions_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        model = kwargs.get("model", "")
        if "ultra" in model.lower():
            raise APITimeoutError("Nebius API request timed out (75s)")
        if call_count <= 2:
            # First two Super attempts return SQL missing a column
            return FakeResponse(content=f"```sql\nCREATE TABLE employees_new (id INT PRIMARY KEY, tag TEXT DEFAULT 'v{call_count}');\nDROP TABLE employees;\nALTER TABLE employees_new RENAME TO employees;\n```")
        # Super fallback succeeds with verified SQL!
        return FakeResponse(content="```sql\nALTER TABLE employees ADD COLUMN notes TEXT DEFAULT '';\n```")

    client.chat.completions.create.side_effect = completions_side_effect

    notifications = []

    def on_progress(msg, data):
        notifications.append(msg)

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Add notes column",
            spec=spec,
            target_table="employees",
            api_key_override="test_key_123",
            max_steps=4,
            progress_callback=on_progress,
            wall_clock_timeout_s=120.0,
        )

    # Must have logged fallback notification instead of fatal AGENT_ERROR
    assert any("Ultra model timed out; falling back to Super model for repair" in n for n in notifications)
    assert result.status in ("VERIFIED", "UNVERIFIED")
    assert "ADD COLUMN notes" in result.up_sql


def test_permanent_llm_timeout_on_all_models_yields_agent_error():
    """When all models (including fallbacks) permanently time out, loop terminates with AGENT_ERROR."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY);")
    conn.commit()

    spec = MigrationSpec(goal="Test", affected_tables=["employees"])

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]
    client.chat.completions.create.side_effect = APITimeoutError("Connection timed out (45s)")

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Test",
            spec=spec,
            target_table="employees",
            api_key_override="test_key_123",
            max_steps=2,
            wall_clock_timeout_s=60.0,
        )

    assert result.status == "FAILED"
    assert "AGENT_ERROR" in result.stop_reason
    assert "timed out" in result.stop_reason
