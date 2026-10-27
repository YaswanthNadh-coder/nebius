"""Mutation corpus tests: >= 15 flawed or malicious migrations, ensuring each is flagged."""

import sqlite3

import pytest

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.verify.invariants import evaluate_migration_invariants
from timemachine.verify.rebuild import verify_rollback_roundtrip
from timemachine.verify.spec import ColumnMapping, MigrationSpec


@pytest.fixture
def base_test_db():
    """Create a baseline database with employees, departments, indexes, views, and triggers."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE departments (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL,
            dept_code TEXT,
            salary_text TEXT,
            created_at TEXT DEFAULT '2026-01-01 12:00:00',
            FOREIGN KEY (dept_code) REFERENCES departments(code)
        );
    """)
    cur.execute("CREATE INDEX idx_emp_dept ON employees(dept_code);")
    cur.execute("CREATE VIEW v_emp_depts AS SELECT e.id, e.full_name, d.name FROM employees e LEFT JOIN departments d ON e.dept_code = d.code;")
    cur.execute("CREATE TRIGGER trg_emp_ins AFTER INSERT ON employees BEGIN SELECT 1; END;")

    cur.executemany(
        "INSERT INTO departments (code, name) VALUES (?, ?);",
        [("ENG", "Engineering"), ("MKT", "Marketing")],
    )
    cur.executemany(
        "INSERT INTO employees (id, full_name, dept_code, salary_text) VALUES (?, ?, ?, ?);",
        [
            (1, "Alice Smith", "ENG", "$95,000"),
            (2, "Bob Jones", "MKT", "$72,000"),
            (3, "Charlie Brown", None, "$68,000"),
            (4, "Diana Prince", "LEGACY_99", "$110,000"),
        ],
    )
    conn.commit()
    return conn


def _run_mutation(base_conn, sql, spec=None, target_table="employees", expected_row_fn=None):
    """Helper to run a migration on a clone and evaluate invariants."""
    clone = sqlite3.connect(":memory:")
    base_conn.backup(clone)
    adapter = SQLiteAdapter()
    ok, err, elapsed = adapter.run_in_sandbox(clone, sql, fk_enabled=True)
    if not ok:
        return False, err

    inv = evaluate_migration_invariants(
        base_conn,
        clone,
        target_table=target_table,
        spec=spec,
        expected_row_fn=expected_row_fn,
        elapsed_ms=elapsed,
    )
    return inv.passed, (inv.failed_checks[0] if inv.failed_checks else None)


# 1. Garbage constants: setting every row to 'x'/'y'/'UNASSIGNED'/0
def test_mutation_garbage_constants(base_test_db):
    sql = """
    DROP VIEW IF EXISTS v_emp_depts;
    CREATE TABLE employees_new (
        id INT PRIMARY KEY,
        first_name TEXT,
        last_name TEXT,
        dept_code TEXT,
        salary_num INT
    );
    INSERT INTO employees_new SELECT id, 'x', 'y', 'UNASSIGNED', 0 FROM employees;
    DROP TABLE employees;
    ALTER TABLE employees_new RENAME TO employees;
    CREATE VIEW v_emp_depts AS SELECT e.id, e.first_name, d.name FROM employees e LEFT JOIN departments d ON e.dept_code = d.code;
    """
    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="first_name", source_column="full_name", transform="split_name_first"),
                ColumnMapping(new_column="last_name", source_column="full_name", transform="split_name_last"),
                ColumnMapping(new_column="dept_code", source_column="dept_code", transform="default_if_null", default_value="UNASSIGNED"),
                ColumnMapping(new_column="salary_num", source_column="salary_text", transform="cast_int"),
            ]
        },
    )
    passed, error = _run_mutation(base_test_db, sql, spec=spec)
    assert passed is False
    assert "DATA_MISMATCH" in error or "first_name" in error


# 2. Wrong default: missing or incorrect default on NOT NULL column
def test_mutation_wrong_default(base_test_db):
    sql = """
    ALTER TABLE employees ADD COLUMN status TEXT;
    """
    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="status", null_policy="not_null", default_value="active"),
            ]
        },
    )
    passed, error = _run_mutation(base_test_db, sql, spec=spec)
    assert passed is False
    assert "SCHEMA_CONSTRAINT_VIOLATION" in error or "NULL_CORRUPTION" in error


# 3. Lost index: rebuild table without recreating pre-existing index
def test_mutation_lost_index(base_test_db):
    sql = """
    DROP VIEW IF EXISTS v_emp_depts;
    CREATE TABLE employees_new (
        id INT PRIMARY KEY,
        full_name TEXT NOT NULL,
        dept_code TEXT,
        salary_text TEXT,
        created_at TEXT
    );
    INSERT INTO employees_new SELECT id, full_name, dept_code, salary_text, created_at FROM employees;
    DROP TABLE employees;
    ALTER TABLE employees_new RENAME TO employees;
    CREATE VIEW v_emp_depts AS SELECT e.id, e.full_name, d.name FROM employees e LEFT JOIN departments d ON e.dept_code = d.code;
    CREATE TRIGGER trg_emp_ins AFTER INSERT ON employees BEGIN SELECT 1; END;
    """
    passed, error = _run_mutation(base_test_db, sql)
    assert passed is False
    assert "LOST_INDEX" in error and "idx_emp_dept" in error


# 4. Lost view / trigger
def test_mutation_lost_view_or_trigger(base_test_db):
    sql = "DROP VIEW v_emp_depts;"
    passed, error = _run_mutation(base_test_db, sql)
    assert passed is False
    assert "LOST_VIEW" in error


# 5. Broken foreign key: references non-existent department
def test_mutation_broken_fk(base_test_db):
    sql = "INSERT INTO employees (id, full_name, dept_code, salary_text) VALUES (99, 'Rogue', 'NON_EXISTENT_DEPT', '$50,000');"
    passed, error = _run_mutation(base_test_db, sql)
    assert passed is False
    assert "FOREIGN KEY" in error.upper() or "FOREIGN_KEY" in error


# 6. Duplicated rows: faulty join causing duplication
def test_mutation_duplicated_rows(base_test_db):
    sql = """
    DROP VIEW IF EXISTS v_emp_depts;
    CREATE TABLE employees_new (
        id INT,
        full_name TEXT,
        dept_code TEXT,
        salary_text TEXT,
        created_at TEXT
    );
    INSERT INTO employees_new SELECT e.id, e.full_name, e.dept_code, e.salary_text, e.created_at FROM employees e CROSS JOIN departments d;
    DROP TABLE employees;
    ALTER TABLE employees_new RENAME TO employees;
    CREATE VIEW v_emp_depts AS SELECT e.id, e.full_name, d.name FROM employees e LEFT JOIN departments d ON e.dept_code = d.code;
    """
    passed, error = _run_mutation(base_test_db, sql)
    assert passed is False
    assert "DUPLICAT" in error or "ROW_COUNT_MISMATCH" in error


# 7. Dropped rows: naive INNER JOIN silently dropping NULL dept_code rows
def test_mutation_dropped_rows(base_test_db):
    sql = """
    DROP VIEW IF EXISTS v_emp_depts;
    CREATE TABLE employees_new (
        id INT PRIMARY KEY,
        full_name TEXT,
        dept_code TEXT,
        salary_text TEXT,
        created_at TEXT
    );
    INSERT INTO employees_new
    SELECT e.id, e.full_name, e.dept_code, e.salary_text, e.created_at
    FROM employees e
    INNER JOIN departments d ON e.dept_code = d.code;
    DROP TABLE employees;
    ALTER TABLE employees_new RENAME TO employees;
    CREATE VIEW v_emp_depts AS SELECT e.id, e.full_name, d.name FROM employees e LEFT JOIN departments d ON e.dept_code = d.code;
    """
    passed, error = _run_mutation(base_test_db, sql)
    assert passed is False
    assert "DATA_LOSS" in error or "ROW_LOSS" in error or "ROW_COUNT_MISMATCH" in error


# 8. Silent coercion: unhandled strings silently cast to 0
def test_mutation_silent_coercion(base_test_db):
    sql = """
    DROP VIEW IF EXISTS v_emp_depts;
    CREATE TABLE employees_new (
        id INT PRIMARY KEY,
        full_name TEXT,
        salary_num INT
    );
    INSERT INTO employees_new SELECT id, full_name, CAST(salary_text AS INTEGER) FROM employees;
    DROP TABLE employees;
    ALTER TABLE employees_new RENAME TO employees;
    CREATE VIEW v_emp_depts AS SELECT e.id, e.full_name, d.name FROM employees e LEFT JOIN departments d ON e.dept_code = d.code;
    """
    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="salary_num", source_column="salary_text", transform="cast_int"),
            ]
        },
    )
    # CAST('$95,000' AS INTEGER) in SQLite evaluates to 0 because '$' is non-numeric!
    passed, error = _run_mutation(base_test_db, sql, spec=spec)
    assert passed is False
    assert "DATA_MISMATCH" in error or "salary_num" in error


# 9. Case-folded or mutated unchanged columns
def test_mutation_case_folded_unchanged_column(base_test_db):
    sql = "UPDATE employees SET full_name = LOWER(full_name);"
    spec = MigrationSpec(
        affected_tables=["employees"],
        columns_declared_unchanged={"employees": ["full_name"]},
    )
    passed, error = _run_mutation(base_test_db, sql, spec=spec)
    assert passed is False
    assert "UNTOUCHED_COLUMN_MUTATED" in error


# 10. Truncated strings
def test_mutation_truncated_strings(base_test_db):
    sql = "UPDATE employees SET full_name = SUBSTR(full_name, 1, 3);"
    spec = MigrationSpec(
        affected_tables=["employees"],
        columns_declared_unchanged={"employees": ["full_name"]},
    )
    passed, error = _run_mutation(base_test_db, sql, spec=spec)
    assert passed is False
    assert "UNTOUCHED_COLUMN_MUTATED" in error


# 11. Timezone shift or timestamp mutation
def test_mutation_timestamp_shift(base_test_db):
    sql = "UPDATE employees SET created_at = '1970-01-01 00:00:00';"
    spec = MigrationSpec(
        affected_tables=["employees"],
        columns_declared_unchanged={"employees": ["created_at"]},
    )
    passed, error = _run_mutation(base_test_db, sql, spec=spec)
    assert passed is False
    assert "UNTOUCHED_COLUMN_MUTATED" in error


# 12. Leftover temp table
def test_mutation_leftover_temp_table(base_test_db):
    sql = "CREATE TABLE employees_new (id INT);"
    passed, error = _run_mutation(base_test_db, sql)
    assert passed is False
    assert "LEFTOVER_TEMP_TABLES" in error


# 13. Lossy rollback
def test_mutation_lossy_rollback(base_test_db):
    up_sql = "ALTER TABLE employees DROP COLUMN salary_text;"
    down_sql = "ALTER TABLE employees ADD COLUMN salary_text TEXT DEFAULT NULL;"
    ok, err, lossy = verify_rollback_roundtrip(base_test_db, up_sql, down_sql, target_table="employees")
    assert ok is False
    assert "ROUNDTRIP_CHECKSUM_MISMATCH" in (err or "")


# 14. Unauthorized DROP of non-target table
def test_mutation_unauthorized_drop_non_target(base_test_db):
    sql = "DROP TABLE departments;"
    passed, error = _run_mutation(base_test_db, sql, target_table="employees")
    assert passed is False
    assert "FOREIGN KEY" in error.upper() or "TABLE_SET_MISMATCH" in error or "NON_TARGET_TABLE" in error


# 15. Sandbox escape ATTACH
def test_mutation_sandbox_escape_attach(base_test_db):
    sql = "ATTACH DATABASE 'rogue.db' AS rogue;"
    passed, error = _run_mutation(base_test_db, sql)
    assert passed is False
    assert "SANDBOX VIOLATION" in error or "authoriz" in error.lower()


# 16. Sandbox escape VACUUM INTO
def test_mutation_sandbox_escape_vacuum(base_test_db):
    sql = "VACUUM INTO 'rogue_dump.db';"
    passed, error = _run_mutation(base_test_db, sql)
    assert passed is False
    assert "SANDBOX VIOLATION" in error or "authoriz" in error.lower()


# 17. Infinite loop / recursion
def test_mutation_infinite_loop(base_test_db):
    sql = """
    WITH RECURSIVE inf(x) AS (
        SELECT 1
        UNION ALL
        SELECT x + 1 FROM inf
    )
    SELECT count(*) FROM inf;
    """
    passed, error = _run_mutation(base_test_db, sql)
    assert passed is False
    assert "TIMEOUT" in error or "interrupted" in error.lower()
