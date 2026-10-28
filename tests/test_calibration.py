"""Tests calibrating the verifier against hand-written and naive migrations for built-in scenarios."""

import sqlite3

from timemachine.agent.planner import build_spec_prompt
from timemachine.agent.prompts import generate_transform_contract
from timemachine.traps.generator import inject_adversarial_traps
from timemachine.ui.app_view import create_builtin_scenario_adapter
from timemachine.verify.invariants import evaluate_migration_invariants
from timemachine.verify.rebuild import verify_rollback_roundtrip
from timemachine.verify.spec import ColumnMapping, MigrationSpec


def test_scenario_1_handwritten_passes_and_naive_fails():
    """Scenario 1: Department Normalization & Name Split.

    - Correct migration with full whitespace handling (including tabs) passes with status VERIFIED
      and rollback_verified is True (EQUIVALENT_AFTER_NORMALIZATION).
    - Naive migration using plain TRIM(full_name) fails with DATA_MISMATCH.
    """
    adapter = create_builtin_scenario_adapter(1)
    base_conn = adapter.clone()
    schema = adapter.introspect()
    inject_adversarial_traps(base_conn, schema)

    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="id", source_column="id", transform="passthrough", null_policy="nullable"),
                ColumnMapping(new_column="first_name", source_column="full_name", transform="split_name_first", null_policy="not_null"),
                ColumnMapping(new_column="last_name", source_column="full_name", transform="split_name_last", default_value="N/A", null_policy="not_null"),
                ColumnMapping(new_column="department", source_column="department", transform="passthrough", null_policy="nullable"),
                ColumnMapping(new_column="salary", source_column="salary", transform="passthrough", null_policy="nullable"),
                ColumnMapping(new_column="created_at", source_column="created_at", transform="passthrough", null_policy="nullable"),
            ]
        },
    )

    # 1. Hand-written correct migration: strips tabs, handles mononyms, recreates index & trigger
    correct_up_sql = """
    DROP TRIGGER IF EXISTS trg_emp_audit;
    CREATE TABLE employees_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        department TEXT,
        salary INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    INSERT INTO employees_new (id, first_name, last_name, department, salary, created_at)
    SELECT
        id,
        CASE
            WHEN INSTR(TRIM(full_name, ' ' || char(9) || char(10) || char(13)), ' ') > 0
            THEN SUBSTR(TRIM(full_name, ' ' || char(9) || char(10) || char(13)), 1, INSTR(TRIM(full_name, ' ' || char(9) || char(10) || char(13)), ' ') - 1)
            ELSE TRIM(full_name, ' ' || char(9) || char(10) || char(13))
        END AS first_name,
        CASE
            WHEN INSTR(TRIM(full_name, ' ' || char(9) || char(10) || char(13)), ' ') > 0
            THEN TRIM(SUBSTR(TRIM(full_name, ' ' || char(9) || char(10) || char(13)), INSTR(TRIM(full_name, ' ' || char(9) || char(10) || char(13)), ' ') + 1), ' ' || char(9) || char(10) || char(13))
            ELSE 'N/A'
        END AS last_name,
        department,
        salary,
        created_at
    FROM employees;

    DROP TABLE employees;
    ALTER TABLE employees_new RENAME TO employees;

    CREATE INDEX idx_emp_salary ON employees(salary);
    CREATE TRIGGER trg_emp_audit AFTER INSERT ON employees
    BEGIN
        SELECT 1;
    END;
    """

    correct_down_sql = """
    DROP TRIGGER IF EXISTS trg_emp_audit;
    CREATE TABLE employees_old (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        department TEXT,
        salary INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    INSERT INTO employees_old (id, full_name, department, salary, created_at)
    SELECT
        id,
        CASE
            WHEN last_name = 'N/A' THEN first_name
            ELSE first_name || ' ' || last_name
        END AS full_name,
        department,
        salary,
        created_at
    FROM employees;

    DROP TABLE employees;
    ALTER TABLE employees_old RENAME TO employees;

    CREATE INDEX idx_emp_salary ON employees(salary);
    CREATE TRIGGER trg_emp_audit AFTER INSERT ON employees
    BEGIN
        SELECT 1;
    END;
    """

    test_conn = sqlite3.connect(":memory:")
    base_conn.backup(test_conn)
    test_conn.executescript(correct_up_sql)

    inv_result = evaluate_migration_invariants(base_conn, test_conn, "employees", spec=spec)
    assert inv_result.status == "VERIFIED", f"Expected VERIFIED, got {inv_result.status}: {inv_result.failed_checks}"
    assert inv_result.passed is True
    assert len(inv_result.failed_checks) == 0

    roundtrip = verify_rollback_roundtrip(base_conn, correct_up_sql, correct_down_sql, "employees")
    assert roundtrip.classification in ("EXACT", "EQUIVALENT_AFTER_NORMALIZATION")
    test_conn.close()

    # 2. Naive migration: uses plain TRIM(full_name), leaving tab characters in trap rows
    naive_up_sql = """
    DROP TRIGGER IF EXISTS trg_emp_audit;
    CREATE TABLE employees_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        department TEXT,
        salary INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    INSERT INTO employees_new (id, first_name, last_name, department, salary, created_at)
    SELECT
        id,
        CASE
            WHEN INSTR(TRIM(full_name), ' ') > 0
            THEN SUBSTR(TRIM(full_name), 1, INSTR(TRIM(full_name), ' ') - 1)
            ELSE TRIM(full_name)
        END AS first_name,
        CASE
            WHEN INSTR(TRIM(full_name), ' ') > 0
            THEN SUBSTR(TRIM(full_name), INSTR(TRIM(full_name), ' ') + 1)
            ELSE 'N/A'
        END AS last_name,
        department,
        salary,
        created_at
    FROM employees;

    DROP TABLE employees;
    ALTER TABLE employees_new RENAME TO employees;

    CREATE INDEX idx_emp_salary ON employees(salary);
    CREATE TRIGGER trg_emp_audit AFTER INSERT ON employees
    BEGIN
        SELECT 1;
    END;
    """

    naive_conn = sqlite3.connect(":memory:")
    base_conn.backup(naive_conn)
    naive_conn.executescript(naive_up_sql)

    naive_result = evaluate_migration_invariants(base_conn, naive_conn, "employees", spec=spec)
    assert naive_result.status == "FAILED"
    assert any("DATA_MISMATCH" in check for check in naive_result.failed_checks)
    naive_conn.close()
    base_conn.close()


def test_scenario_2_handwritten_passes_and_naive_fails():
    """Scenario 2: Legacy Salary Banding & Normalization.

    - Correct migration parsing salary cents and salary band with full trim passes with VERIFIED.
    - Naive migration using plain TRIM without tab handling fails with DATA_MISMATCH.
    """
    adapter = create_builtin_scenario_adapter(2)
    base_conn = adapter.clone()
    schema = adapter.introspect()
    inject_adversarial_traps(base_conn, schema)

    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="id", source_column="id", transform="passthrough", null_policy="nullable"),
                ColumnMapping(new_column="name", source_column="name", transform="trim", null_policy="not_null"),
                ColumnMapping(new_column="salary_cents", source_column="raw_salary", transform="cast_cents", null_policy="not_null"),
                ColumnMapping(new_column="salary_band", source_column="raw_salary", transform="salary_band", null_policy="not_null"),
                ColumnMapping(new_column="bonus_pct", source_column="bonus_pct", transform="passthrough", null_policy="nullable"),
                ColumnMapping(new_column="dept_code", source_column="dept_code", transform="passthrough", null_policy="nullable"),
            ]
        },
    )

    correct_up_sql = """
    CREATE TABLE employees_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        salary_cents INTEGER NOT NULL,
        salary_band TEXT NOT NULL,
        bonus_pct TEXT,
        dept_code TEXT
    );

    INSERT INTO employees_new (id, name, salary_cents, salary_band, bonus_pct, dept_code)
    SELECT
        id,
        TRIM(name, ' ' || char(9) || char(10) || char(13)) AS name,
        CAST(ROUND(CAST(REPLACE(REPLACE(REPLACE(TRIM(raw_salary, ' ' || char(9) || char(10) || char(13)), '$', ''), ',', ''), ' ', '') AS REAL) * 100) AS INTEGER) AS salary_cents,
        CASE
            WHEN CAST(REPLACE(REPLACE(REPLACE(TRIM(raw_salary, ' ' || char(9) || char(10) || char(13)), '$', ''), ',', ''), ' ', '') AS REAL) >= 100000 THEN 'HIGH'
            WHEN CAST(REPLACE(REPLACE(REPLACE(TRIM(raw_salary, ' ' || char(9) || char(10) || char(13)), '$', ''), ',', ''), ' ', '') AS REAL) >= 60000 THEN 'MID'
            ELSE 'LOW'
        END AS salary_band,
        bonus_pct,
        dept_code
    FROM employees;

    DROP TABLE employees;
    ALTER TABLE employees_new RENAME TO employees;
    """

    test_conn = sqlite3.connect(":memory:")
    base_conn.backup(test_conn)
    test_conn.executescript(correct_up_sql)

    inv_result = evaluate_migration_invariants(base_conn, test_conn, "employees", spec=spec)
    assert inv_result.status == "VERIFIED", f"Expected VERIFIED, got {inv_result.status}: {inv_result.failed_checks}"
    assert inv_result.passed is True
    assert len(inv_result.failed_checks) == 0
    test_conn.close()

    # Naive migration: plain TRIM(name) leaves \t on trap row
    naive_up_sql = """
    CREATE TABLE employees_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        salary_cents INTEGER NOT NULL,
        salary_band TEXT NOT NULL,
        bonus_pct TEXT,
        dept_code TEXT
    );

    INSERT INTO employees_new (id, name, salary_cents, salary_band, bonus_pct, dept_code)
    SELECT
        id,
        TRIM(name) AS name,
        CAST(ROUND(CAST(REPLACE(REPLACE(raw_salary, '$', ''), ',', '') AS REAL) * 100) AS INTEGER) AS salary_cents,
        CASE
            WHEN CAST(REPLACE(REPLACE(raw_salary, '$', ''), ',', '') AS REAL) >= 100000 THEN 'HIGH'
            WHEN CAST(REPLACE(REPLACE(raw_salary, '$', ''), ',', '') AS REAL) >= 60000 THEN 'MID'
            ELSE 'LOW'
        END AS salary_band,
        bonus_pct,
        dept_code
    FROM employees;

    DROP TABLE employees;
    ALTER TABLE employees_new RENAME TO employees;
    """

    naive_conn = sqlite3.connect(":memory:")
    base_conn.backup(naive_conn)
    naive_conn.executescript(naive_up_sql)

    naive_result = evaluate_migration_invariants(base_conn, naive_conn, "employees", spec=spec)
    assert naive_result.status == "FAILED"
    assert any("DATA_MISMATCH" in check for check in naive_result.failed_checks)
    naive_conn.close()
    base_conn.close()


def test_whitespace_agreement_in_prompts():
    """Plan prompt and transform contract agree with oracle Python str.strip() behavior."""
    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="first_name", source_column="full_name", transform="split_name_first"),
                ColumnMapping(new_column="last_name", source_column="full_name", transform="split_name_last"),
                ColumnMapping(new_column="salary_cents", source_column="raw_salary", transform="cast_cents"),
            ]
        },
    )
    contract = generate_transform_contract(spec)
    assert "str.strip()" in contract

    prompt = build_spec_prompt(schema=create_builtin_scenario_adapter(1).introspect(), goal="Split names")
    assert "str.strip()" in prompt
