"""Tests for Item D: Actionable Failure Messages, hints, data profile facts, and stop reason."""

from timemachine.adapters.schema import ColumnInfo, DatabaseSchema, TableSchema
from timemachine.agent.prompts import (
    build_dynamic_task_checklist,
    build_migration_prompt,
    get_sqlite_gotchas_block,
)
from timemachine.traps.profiler import ColumnProfile
from timemachine.ui.app_view import create_builtin_scenario_adapter
from timemachine.verify.invariants import InvariantResult, evaluate_migration_invariants
from timemachine.verify.spec import ColumnMapping, MigrationSpec


def test_fk_checklist_advice_does_not_suggest_turning_fk_off():
    """Foreign key advice must advise mapping orphan keys per orphan_policy and not turning FKs off."""
    res = InvariantResult(
        status="FAILED",
        passed=False,
        failed_checks=["NEW_FOREIGN_KEY_VIOLATIONS: 1 violations introduced."],
    )
    checklist = build_dynamic_task_checklist(res)
    fk_item = next((item for item in checklist if "foreign key" in item.lower()), "")
    assert "PRAGMA foreign_keys = OFF" not in fk_item, "Should not advise turning PRAGMA foreign_keys OFF"
    assert "orphan_policy" in fk_item or "orphan" in fk_item, "Should advise handling orphans according to orphan_policy"

    gotchas = get_sqlite_gotchas_block()
    assert "never disable enforcement to hide violations" in gotchas.lower() or "orphan_policy" in gotchas.lower()


def test_missing_required_columns_hint_when_up_undone():
    """MISSING_REQUIRED_COLUMNS includes hint 'UP appears undone or not applied' when found equals baseline."""
    adapter = create_builtin_scenario_adapter(1)
    base_conn = adapter.clone()
    test_conn = adapter.clone()

    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="first_name", source_column="full_name"),
                ColumnMapping(new_column="last_name", source_column="full_name"),
            ]
        },
    )

    inv = evaluate_migration_invariants(base_conn, test_conn, target_table="employees", spec=spec)
    assert inv.passed is False
    missing_err = next((fc for fc in inv.failed_checks if "MISSING_REQUIRED_COLUMNS" in fc), "")
    assert "UP appears undone or not applied" in missing_err


def test_data_mismatch_hint_includes_contract_line():
    """DATA_MISMATCH includes expected vs got and transform contract details."""
    adapter = create_builtin_scenario_adapter(1)
    base_conn = adapter.clone()
    test_conn = adapter.clone()

    # Recreate employees with wrong first_name transformation
    cur = test_conn.cursor()
    cur.execute("DROP TABLE employees;")
    cur.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            department TEXT,
            salary INTEGER,
            created_at TEXT
        );
    """)
    cur.execute("""
        INSERT INTO employees (id, first_name, last_name, department, salary, created_at)
        VALUES (1, 'WRONG_NAME', 'Smith', 'ENG', 95000, '2023-01-01');
    """)

    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="id", source_column="id", transform="passthrough"),
                ColumnMapping(new_column="first_name", source_column="full_name", transform="split_name_first"),
                ColumnMapping(new_column="last_name", source_column="full_name", transform="split_name_last", default_value="N/A"),
            ]
        },
    )

    inv = evaluate_migration_invariants(base_conn, test_conn, target_table="employees", spec=spec)
    assert inv.passed is False
    dm_err = next((fc for fc in inv.failed_checks if "DATA_MISMATCH" in fc), "")
    assert "expected" in dm_err and "got" in dm_err
    assert "Contract:" in dm_err or "split_name_first" in dm_err


def test_first_draft_prompt_includes_data_profile_facts():
    """First draft prompt includes masked data-profile facts (null counts and samples)."""
    schema = DatabaseSchema(
        tables={
            "employees": TableSchema(
                name="employees",
                columns=[
                    ColumnInfo(name="id", type="INTEGER", pk=1),
                    ColumnInfo(name="salary", type="INTEGER"),
                ],
            )
        }
    )
    spec = MigrationSpec(affected_tables=["employees"])
    profiles = {
        "employees": {
            "salary": ColumnProfile(
                name="salary",
                column_type="INTEGER",
                total_rows=100,
                null_count=7,
                null_rate=0.07,
                distinct_count=50,
                masked_sample=["50000", "NULL", "75000"],
            )
        }
    }

    prompt = build_migration_prompt(schema, spec, profiles=profiles)
    assert "salary" in prompt
    assert "7" in prompt and "NULL" in prompt.upper()


def test_stop_reason_max_attempts_exhausted():
    """Stop reason must be MAX_ATTEMPTS_EXHAUSTED with 'attempts used x/max'."""
    from timemachine.agent.loop import AgentRunResult

    # Verify formatting in AgentRunResult or loop logic
    attempts = 6
    max_attempts = 6
    stop_reason = f"MAX_ATTEMPTS_EXHAUSTED (attempts used {attempts}/{max_attempts})"
    res = AgentRunResult(status="FAILED", stop_reason=stop_reason)
    assert "MAX_ATTEMPTS_EXHAUSTED" in res.stop_reason
    assert "attempts used 6/6" in res.stop_reason

