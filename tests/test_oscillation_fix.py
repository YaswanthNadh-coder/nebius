"""Tests for eliminating oscillation loops, foreign key diagnostics, and best-attempt ranking."""

import sqlite3

from timemachine.adapters.schema import ColumnInfo, DatabaseSchema, TableSchema
from timemachine.agent.loop import strip_thinking
from timemachine.agent.planner import auto_fix_spec_null_policies
from timemachine.verify.invariants import (
    InvariantResult,
    audit_crash_diagnostics,
    evaluate_migration_invariants,
)
from timemachine.verify.spec import ColumnMapping, MigrationSpec


def test_pk_column_satisfies_not_null():
    """SQLite INTEGER PRIMARY KEY satisfies NOT NULL in invariant verification."""
    base_conn = sqlite3.connect(":memory:")
    base_conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT);")
    base_conn.execute("INSERT INTO employees (full_name) VALUES ('Alice');")
    base_conn.commit()

    test_conn = sqlite3.connect(":memory:")
    # In SQLite, INTEGER PRIMARY KEY without explicit NOT NULL reports notnull=0 in PRAGMA table_info
    test_conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT);")
    test_conn.execute("INSERT INTO employees (id, full_name) VALUES (1, 'Alice');")
    test_conn.commit()

    spec = MigrationSpec(
        goal="Test PK nullability",
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="id", source_column="id", transform="passthrough", null_policy="not_null"),
                ColumnMapping(new_column="full_name", source_column="full_name", transform="passthrough", null_policy="nullable"),
            ]
        },
    )

    res = evaluate_migration_invariants(base_conn, test_conn, target_table="employees", spec=spec)
    assert not any("SCHEMA_CONSTRAINT_VIOLATION: Column 'id' is not defined as NOT NULL" in fc for fc in res.failed_checks)
    assert "PRIMARY_KEY_DECLARED" in res.passed_checks
    base_conn.close()
    test_conn.close()


def test_spec_unchanged_column_cannot_be_not_null():
    """Unchanged columns that were nullable in source must not be forced NOT NULL in spec."""
    spec = MigrationSpec(
        goal="Split name",
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="salary", source_column="salary", transform="passthrough", null_policy="not_null"),
            ]
        },
        columns_declared_unchanged={"employees": ["salary"]},
    )
    # 1. get_non_null_columns excludes unchanged passthrough columns
    nn_cols = spec.get_non_null_columns("employees")
    assert "salary" not in nn_cols

    # 2. auto_fix_spec_null_policies mutates null_policy to nullable
    schema = DatabaseSchema(
        tables={
            "employees": TableSchema(
                name="employees",
                columns=[
                    ColumnInfo(name="id", type="INTEGER", notnull=False, default=None, pk=1),
                    ColumnInfo(name="salary", type="INTEGER", notnull=False, default=None, pk=0),
                ],
                row_count=1,
            )
        }
    )
    auto_fix_spec_null_policies(spec, schema)
    mapping = next(m for m in spec.column_mappings["employees"] if m.new_column == "salary")
    assert mapping.null_policy == "nullable"


def test_audit_crash_diagnostics_foreign_key():
    """When execution crashes on a FOREIGN KEY constraint, diagnostic provides parent keys, orphan values, and SQL recipe."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE departments (dept_code TEXT PRIMARY KEY, name TEXT);")
    conn.execute("INSERT INTO departments VALUES ('ENG', 'Engineering'), ('HR', 'Human Resources');")
    conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, dept_code TEXT);")
    conn.execute("INSERT INTO employees VALUES (1, 'ENG'), (2, 'LEGACY_99'), (3, NULL);")
    conn.commit()

    err_msg = "SQLITE EXECUTION CRASH: FOREIGN KEY constraint failed"
    current_sql = """
    CREATE TABLE employees_new (
        id INTEGER PRIMARY KEY,
        dept_code TEXT REFERENCES departments(dept_code)
    );
    INSERT INTO employees_new SELECT id, dept_code FROM employees;
    """

    diag = audit_crash_diagnostics(conn, "employees", err_msg, current_sql=current_sql)
    assert "[FOREIGN KEY CONSTRAINT CRASH DIAGNOSTIC" in diag
    assert "'departments(dept_code)'" in diag
    assert "'ENG'" in diag
    assert "'LEGACY_99'" in diag
    assert "CONCRETE REMEDIATION RECIPE" in diag
    assert "CASE" in diag
    assert "departments" in diag
    conn.close()


def test_strip_thinking_removes_reasoning_blocks():
    """strip_thinking properly strips closed and unclosed <think> blocks."""
    raw = "<think>\nThinking about foreign keys...\nLet's analyze schema.\n</think>\n```sql\nSELECT 1;\n```"
    cleaned = strip_thinking(raw)
    assert cleaned == "```sql\nSELECT 1;\n```"

    unclosed = "<think>\nStill reasoning without closing tag"
    cleaned_unclosed = strip_thinking(unclosed)
    assert cleaned_unclosed == ""


def test_best_attempt_ranking_logic():
    """Non-crashes (passed_checks > 0) strictly outrank execution crashes."""
    # Attempt A: 23 passed, 2 failed (non-crash)
    inv_a = InvariantResult(status="FAILED", passed=False, passed_checks=[f"P{i}" for i in range(23)], failed_checks=["F1", "F2"])
    # Attempt B: 0 passed, 1 failed (execution crash)
    inv_b = InvariantResult(status="FAILED", passed=False, passed_checks=[], failed_checks=["CRASH"])

    def is_better(new_inv, new_run_ok, best_inv, best_run_ok):
        new_is_non_crash = (new_run_ok and len(new_inv.passed_checks) > 0)
        best_is_non_crash = (best_run_ok and len(best_inv.passed_checks) > 0) if best_inv else False
        new_fc = len(new_inv.failed_checks)
        best_fc = len(best_inv.failed_checks) if best_inv else 999999
        new_pc = len(new_inv.passed_checks)
        best_pc = len(best_inv.passed_checks) if best_inv else -1

        if new_is_non_crash and not best_is_non_crash:
            return True
        if new_is_non_crash == best_is_non_crash:
            if new_fc < best_fc:
                return True
            if new_fc == best_fc and new_pc > best_pc:
                return True
        return False

    # First attempt is B (crash)
    assert is_better(inv_b, False, None, False) is True
    # Second attempt is A (non-crash): should replace B even though len(failed_checks) is 2 > 1
    assert is_better(inv_a, True, inv_b, False) is True
    # Third attempt is B again (crash): should NOT replace A
    assert is_better(inv_b, False, inv_a, True) is False
