"""Tests for Item D: Rollback Verification (rebuild.py verify_rollback_roundtrip classification and diffs)."""

import sqlite3

from timemachine.verify.rebuild import verify_rollback_roundtrip


def _setup_employees_db(with_dirty_whitespace: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL,
            department TEXT,
            salary INTEGER
        );
    """)
    if with_dirty_whitespace:
        conn.executemany(
            "INSERT INTO employees (id, full_name, department, salary) VALUES (?, ?, ?, ?);",
            [
                (1, "  Alice Smith  ", "ENG", 95000),
                (2, "Bob Jones\t\n", "MKT", 72000),
                (3, "Charlie Brown", "HR", 68000),
            ],
        )
    else:
        conn.executemany(
            "INSERT INTO employees (id, full_name, department, salary) VALUES (?, ?, ?, ?);",
            [
                (1, "Alice Smith", "ENG", 95000),
                (2, "Bob Jones", "MKT", 72000),
                (3, "Charlie Brown", "HR", 68000),
            ],
        )
    conn.commit()
    return conn


def test_correct_pair_with_trap_rows_returns_equivalent_after_normalization():
    """Correct oracle-passing UP/DOWN pair with trap rows returns EQUIVALENT_AFTER_NORMALIZATION."""
    conn = _setup_employees_db(with_dirty_whitespace=True)

    up_sql = """
    CREATE TABLE employees_new (
        id INTEGER PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        department TEXT,
        salary INTEGER
    );
    INSERT INTO employees_new (id, first_name, last_name, department, salary)
    SELECT
        id,
        TRIM(SUBSTR(TRIM(full_name, ' ' || char(9) || char(10) || char(13)), 1,
             INSTR(TRIM(full_name, ' ' || char(9) || char(10) || char(13)) || ' ', ' ') - 1)),
        TRIM(SUBSTR(TRIM(full_name, ' ' || char(9) || char(10) || char(13)),
             INSTR(TRIM(full_name, ' ' || char(9) || char(10) || char(13)) || ' ', ' ') + 1)),
        department,
        salary
    FROM employees;
    DROP TABLE employees;
    ALTER TABLE employees_new RENAME TO employees;
    """

    down_sql = """
    CREATE TABLE employees_old (
        id INTEGER PRIMARY KEY,
        full_name TEXT NOT NULL,
        department TEXT,
        salary INTEGER
    );
    INSERT INTO employees_old (id, full_name, department, salary)
    SELECT
        id,
        first_name || ' ' || last_name,
        department,
        salary
    FROM employees;
    DROP TABLE employees;
    ALTER TABLE employees_old RENAME TO employees;
    """

    res = verify_rollback_roundtrip(conn, up_sql, down_sql, target_table="employees")
    assert res["classification"] == "EQUIVALENT_AFTER_NORMALIZATION"
    assert res["lossy_columns"] == []
    assert res["rollback_verified"] is True
    # Can also unpack as 3-tuple
    ok, err, lossy = res
    assert ok is True
    assert err is None
    assert lossy == []


def test_down_that_drops_a_column_returns_broken():
    """DOWN migration that drops a column returns BROKEN and names the column."""
    conn = _setup_employees_db(with_dirty_whitespace=False)

    up_sql = """
    ALTER TABLE employees ADD COLUMN first_name TEXT;
    """

    # DOWN incorrectly drops salary instead of first_name
    down_sql = """
    CREATE TABLE employees_broken (
        id INTEGER PRIMARY KEY,
        full_name TEXT NOT NULL,
        department TEXT
    );
    INSERT INTO employees_broken (id, full_name, department)
    SELECT id, full_name, department FROM employees;
    DROP TABLE employees;
    ALTER TABLE employees_broken RENAME TO employees;
    """

    res = verify_rollback_roundtrip(conn, up_sql, down_sql, target_table="employees")
    assert res["classification"] == "BROKEN"
    assert "salary" in res["lossy_columns"]
    assert res["rollback_verified"] is False
    assert res.error is not None
    assert "salary" in str(res.error) or "salary" in res["lossy_columns"]


def test_down_that_rewrites_salaries_returns_lossy_with_sample_diffs():
    """DOWN migration that rewrites data (e.g. doubles salary) returns LOSSY with sample_diffs."""
    conn = _setup_employees_db(with_dirty_whitespace=False)

    up_sql = "ALTER TABLE employees ADD COLUMN bonus INTEGER DEFAULT 0;"
    down_sql = """
    ALTER TABLE employees DROP COLUMN bonus;
    UPDATE employees SET salary = salary * 2;
    """

    res = verify_rollback_roundtrip(conn, up_sql, down_sql, target_table="employees")
    assert res["classification"] == "LOSSY"
    assert "salary" in res["lossy_columns"]
    assert res["rollback_verified"] is False
    assert len(res["sample_diffs"]) > 0
    assert len(res["sample_diffs"]) <= 3
    sample = res["sample_diffs"][0]
    assert "pk" in sample
    assert sample["column"] == "salary"
    assert sample["original"] != sample["restored"]
