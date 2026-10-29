"""Built-in scenarios, reference migrations, specs, and satisfiability validation."""

import sqlite3

from timemachine.adapters.base import DbAdapter
from timemachine.verify.invariants import InvariantResult, evaluate_migration_invariants
from timemachine.verify.spec import ColumnMapping, MigrationSpec

# ── Scenario 1 Reference ───────────────────────────────────────────────────────

SCENARIO_1_SPEC = MigrationSpec(
    goal="Split full_name into first_name/last_name and add a foreign key department_code -> departments(dept_code); orphan codes and NULLs become NULL.",
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
    columns_declared_unchanged={"employees": ["id", "department", "salary", "created_at"]},
)

SCENARIO_1_REFERENCE_UP = """
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

SCENARIO_1_REFERENCE_DOWN = """
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

# ── Scenario 2 Reference ───────────────────────────────────────────────────────

SCENARIO_2_SPEC = MigrationSpec(
    goal="Clean raw_salary by converting string amounts ($120,000) to integer cents in salary_cents, and compute salary_band.",
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

SCENARIO_2_REFERENCE_UP = """
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

SCENARIO_2_REFERENCE_DOWN = """
CREATE TABLE employees_old (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    raw_salary TEXT NOT NULL,
    bonus_pct TEXT,
    dept_code TEXT
);

INSERT INTO employees_old (id, name, raw_salary, bonus_pct, dept_code)
SELECT
    id,
    name,
    '$' || printf('%,d', salary_cents / 100),
    bonus_pct,
    dept_code
FROM employees;

DROP TABLE employees;
ALTER TABLE employees_old RENAME TO employees;
"""


def get_builtin_scenario_reference(scenario_id: int) -> tuple[str, str, MigrationSpec]:
    """Return (reference_up, reference_down, spec) for a built-in demo scenario."""
    if scenario_id == 1:
        return SCENARIO_1_REFERENCE_UP, SCENARIO_1_REFERENCE_DOWN, SCENARIO_1_SPEC
    elif scenario_id == 2:
        return SCENARIO_2_REFERENCE_UP, SCENARIO_2_REFERENCE_DOWN, SCENARIO_2_SPEC
    raise ValueError(f"Unknown scenario ID: {scenario_id}")


def validate_spec_satisfiability(
    spec: MigrationSpec,
    adapter: DbAdapter,
    reference_up: str,
    target_table: str = "employees",
) -> InvariantResult:
    """Run reference rebuild in sandbox to verify whether spec itself is satisfiable before calling LLM."""
    base_clone = adapter.clone()
    test_clone = sqlite3.connect(":memory:", check_same_thread=False)
    base_clone.backup(test_clone)
    try:
        ok, err, el = adapter.run_in_sandbox(test_clone, reference_up, fk_enabled=True)
        if not ok:
            return InvariantResult(
                status="FAILED",
                passed=False,
                error=err,
                failed_checks=[f"REFERENCE_EXEC_FAILED: {err}"],
            )
        return evaluate_migration_invariants(base_clone, test_clone, target_table=target_table, spec=spec, elapsed_ms=el)
    finally:
        base_clone.close()
        test_clone.close()
