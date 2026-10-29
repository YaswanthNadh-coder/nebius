"""Tests for eliminating oscillation loops, foreign key diagnostics, and best-attempt ranking."""

import sqlite3

from timemachine.adapters.schema import ColumnInfo, DatabaseSchema, TableSchema
from timemachine.agent.loop import strip_thinking
from timemachine.agent.planner import auto_fix_spec_null_policies
from timemachine.agent.prompts import format_previous_attempts_block
from timemachine.verify.invariants import (
    InvariantResult,
    audit_crash_diagnostics,
    evaluate_migration_invariants,
)
from timemachine.verify.oracle import verify_rows_with_oracle
from timemachine.verify.rebuild import sanitize_down_script, verify_rollback_roundtrip
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


def test_audit_crash_diagnostics_foreign_key_sentinel_recipe():
    """Diagnostic provides 2-step Sentinel Foreign Key Pattern when foreign key fails."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE departments (dept_code TEXT PRIMARY KEY, dept_name TEXT NOT NULL);")
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

    spec = MigrationSpec(
        goal="Map invalid codes to UNASSIGNED",
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="dept_code", source_column="dept_code", transform="map_via_lookup", default_value="UNASSIGNED"),
            ]
        },
    )

    diag = audit_crash_diagnostics(conn, "employees", err_msg, current_sql=current_sql, spec=spec)
    assert "2-STEP SENTINEL FOREIGN KEY PATTERN" in diag or "SENTINEL" in diag.upper()
    assert "INSERT OR IGNORE INTO departments" in diag
    assert "'UNASSIGNED'" in diag
    assert "CASE WHEN" in diag
    assert "departments(dept_code)" in diag
    conn.close()


def test_data_mismatch_with_foreign_key_adds_critical_hint():
    """DATA_MISMATCH failure on a column with a FOREIGN KEY appends explicit sentinel seeding hint."""
    base_conn = sqlite3.connect(":memory:")
    base_conn.execute("CREATE TABLE departments (dept_code TEXT PRIMARY KEY, dept_name TEXT NOT NULL);")
    base_conn.execute("INSERT INTO departments VALUES ('ENG', 'Engineering'), ('HR', 'Human Resources');")
    base_conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, full_name TEXT, department TEXT);")
    base_conn.execute("INSERT INTO employees VALUES (1, 'Alice', 'ENG'), (2, 'Bob', 'LEGACY_99'), (3, 'Charlie', NULL);")
    base_conn.commit()

    # Migrated clone where Bob and Charlie got NULL instead of UNASSIGNED
    mig_conn = sqlite3.connect(":memory:")
    mig_conn.execute("CREATE TABLE departments (dept_code TEXT PRIMARY KEY, dept_name TEXT NOT NULL);")
    mig_conn.execute("INSERT INTO departments VALUES ('ENG', 'Engineering'), ('HR', 'Human Resources');")
    mig_conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, full_name TEXT, department_code TEXT REFERENCES departments(dept_code));")
    mig_conn.execute("INSERT INTO employees VALUES (1, 'Alice', 'ENG'), (2, 'Bob', NULL), (3, 'Charlie', NULL);")
    mig_conn.commit()

    spec = MigrationSpec(
        goal="Normalize department",
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="id", source_column="id", transform="passthrough"),
                ColumnMapping(new_column="full_name", source_column="full_name", transform="passthrough"),
                ColumnMapping(
                    new_column="department_code",
                    source_column="department",
                    transform="map_via_lookup",
                    lookup_table="departments",
                    lookup_key="dept_code",
                    lookup_val="dept_code",
                    default_value="UNASSIGNED",
                ),
            ]
        },
    )

    res = evaluate_migration_invariants(base_conn, mig_conn, target_table="employees", spec=spec)
    assert res.passed is False
    mismatch_fc = next((fc for fc in res.failed_checks if "DATA_MISMATCH" in fc), "")
    assert mismatch_fc != ""
    assert "CRITICAL FK NOTE" in mismatch_fc
    assert "departments(dept_code)" in mismatch_fc
    assert "INSERT OR IGNORE INTO departments" in mismatch_fc
    assert "'UNASSIGNED'" in mismatch_fc
    base_conn.close()
    mig_conn.close()


def test_non_target_parent_table_sentinel_seeding_permitted():
    """Seeding a sentinel row into parent table departments does NOT fail NON_TARGET_TABLE_MUTATED."""
    base_conn = sqlite3.connect(":memory:")
    base_conn.execute("CREATE TABLE departments (dept_code TEXT PRIMARY KEY, dept_name TEXT NOT NULL);")
    base_conn.execute("INSERT INTO departments VALUES ('ENG', 'Engineering'), ('HR', 'Human Resources');")
    base_conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, full_name TEXT, dept_code TEXT);")
    base_conn.execute("INSERT INTO employees VALUES (1, 'Alice', 'ENG');")
    base_conn.commit()

    # Migrated clone: departments has 1 extra row ('UNASSIGNED'), employees is migrated
    mig_conn = sqlite3.connect(":memory:")
    mig_conn.execute("CREATE TABLE departments (dept_code TEXT PRIMARY KEY, dept_name TEXT NOT NULL);")
    mig_conn.execute("INSERT INTO departments VALUES ('ENG', 'Engineering'), ('HR', 'Human Resources'), ('UNASSIGNED', 'Unassigned');")
    mig_conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, full_name TEXT, dept_code TEXT REFERENCES departments(dept_code));")
    mig_conn.execute("INSERT INTO employees VALUES (1, 'Alice', 'ENG');")
    mig_conn.commit()

    spec = MigrationSpec(
        goal="Normalize department",
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="id", source_column="id", transform="passthrough"),
                ColumnMapping(new_column="full_name", source_column="full_name", transform="passthrough"),
                ColumnMapping(new_column="dept_code", source_column="dept_code", transform="passthrough"),
            ]
        },
    )

    res = evaluate_migration_invariants(base_conn, mig_conn, target_table="employees", spec=spec)
    # Check that NON_TARGET_TABLE_MUTATED did NOT fail for departments
    assert not any("NON_TARGET_TABLE_MUTATED: Table 'departments'" in fc for fc in res.failed_checks)
    assert any("NON_TARGET_TABLE_UNTOUCHED_departments" in pc for pc in res.passed_checks)
    base_conn.close()
    mig_conn.close()


def test_sanitize_down_script_relaxes_baseline_nullable_columns():
    """sanitize_down_script relaxes NOT NULL on columns that were nullable in baseline."""
    raw_down = """
    CREATE TABLE employees_old (
        id INTEGER PRIMARY KEY,
        full_name TEXT NOT NULL,
        department TEXT,
        salary INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    INSERT INTO employees_old SELECT id, full_name, department, salary, created_at FROM employees;
    DROP TABLE employees;
    ALTER TABLE employees_old RENAME TO employees;
    """
    nullable_cols = ["salary", "department", "created_at"]
    sanitized = sanitize_down_script(raw_down, nullable_cols)
    assert "salary INTEGER," in sanitized or "salary INTEGER\n" in sanitized
    assert "salary INTEGER NOT NULL" not in sanitized
    assert "created_at TEXT DEFAULT CURRENT_TIMESTAMP" in sanitized
    assert "created_at TEXT NOT NULL" not in sanitized
    # full_name was NOT nullable in baseline, so NOT NULL must be preserved
    assert "full_name TEXT NOT NULL" in sanitized


def test_verify_rollback_auto_relaxes_unwarranted_not_null():
    """verify_rollback_roundtrip auto-relaxes unwarranted NOT NULL constraints on baseline-nullable columns to prevent crashes on NULL rows."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, full_name TEXT NOT NULL, department TEXT, salary INTEGER);")
    conn.execute("INSERT INTO employees VALUES (1, 'Alice Smith', 'ENG', NULL);")
    conn.commit()

    up_sql = "ALTER TABLE employees ADD COLUMN active INTEGER DEFAULT 1;"
    # DOWN script contains a buggy NOT NULL on salary, which would otherwise crash on the NULL row
    down_sql = """
    CREATE TABLE employees_old (
        id INTEGER PRIMARY KEY,
        full_name TEXT NOT NULL,
        department TEXT,
        salary INTEGER NOT NULL
    );
    INSERT INTO employees_old (id, full_name, department, salary) SELECT id, full_name, department, salary FROM employees;
    DROP TABLE employees;
    ALTER TABLE employees_old RENAME TO employees;
    """

    res = verify_rollback_roundtrip(conn, up_sql, down_sql, target_table="employees")
    assert res.rollback_verified is True
    assert res.classification in ("EXACT", "EQUIVALENT_AFTER_NORMALIZATION")
    conn.close()


def test_format_previous_attempts_block_detects_oscillation():
    """format_previous_attempts_block detects oscillation between FOREIGN KEY crash and DATA MISMATCH."""
    from types import SimpleNamespace

    record_fk = SimpleNamespace(
        attempt_num=1,
        model="super",
        approach_summary="Added FK check",
        failed_check_types=["FOREIGN KEY constraint failed"],
        error_message="FOREIGN KEY constraint failed",
    )
    record_dm = SimpleNamespace(
        attempt_num=2,
        model="super",
        approach_summary="Set orphan keys to NULL",
        failed_check_types=["DATA_MISMATCH: department_code wrong"],
        error_message="DATA_MISMATCH: expected UNASSIGNED got None",
    )
    history = SimpleNamespace(
        records=[record_fk, record_dm],
        get_discovered_requirements=lambda: [],
    )

    block = format_previous_attempts_block(history)
    assert "REPAIR OSCILLATION DETECTED BETWEEN FOREIGN KEY CONSTRAINT AND SENTINEL DATA MISMATCH" in block
    assert "2-step Sentinel Foreign Key Pattern" in block
    assert "INSERT OR IGNORE INTO parent_table" in block


def test_format_previous_attempts_block_detects_not_null_oscillation():
    """format_previous_attempts_block detects oscillation between NOT NULL crash and untouched column."""
    from types import SimpleNamespace

    record_nn = SimpleNamespace(
        attempt_num=1,
        model="super",
        approach_summary="Added salary not null",
        failed_check_types=["SQLITE EXECUTION CRASH: NOT NULL constraint failed: salary"],
        error_message="NOT NULL constraint failed: salary",
    )
    record_uc = SimpleNamespace(
        attempt_num=2,
        model="ultra",
        approach_summary="Used coalesce to prevent nulls",
        failed_check_types=["UNTOUCHED_COLUMN_MUTATED: Column 'salary' declared unchanged was modified"],
        error_message="UNTOUCHED_COLUMN_MUTATED",
    )
    history = SimpleNamespace(
        records=[record_nn, record_uc],
        get_discovered_requirements=lambda: [],
    )

    block = format_previous_attempts_block(history)
    assert "REPAIR OSCILLATION DETECTED BETWEEN NOT NULL EXECUTION CRASH AND UNTOUCHED COLUMN PRESERVATION" in block


def test_oracle_lookup_orphan_policy_null_maps_to_none():
    """When orphan_policy is 'null' and default_value is None, map_via_lookup maps unmapped codes to None."""
    orig_rows = [
        {"id": 1, "dept": "ENG"},
        {"id": 2, "dept": "LEGACY_99"},
        {"id": 3, "dept": None},
    ]
    migrated_by_pk = {
        "1": {"id": 1, "department_code": "ENG"},
        "2": {"id": 2, "department_code": None},
        "3": {"id": 3, "department_code": None},
    }
    mapping = ColumnMapping(
        new_column="department_code",
        source_column="dept",
        transform="map_via_lookup",
        lookup_table="departments",
        lookup_key="dept_code",
        lookup_val="dept_code",
        orphan_policy="null",
        default_value=None,
    )
    lookup_data = {"departments": {"ENG": "ENG", "MKT": "MKT"}}

    matched, mismatches, examples = verify_rows_with_oracle(
        orig_rows,
        migrated_by_pk,
        [mapping],
        lookup_data=lookup_data,
        pk_column="id",
    )
    assert matched is True
    assert mismatches == {}
    assert examples == []



