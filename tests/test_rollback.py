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


def test_rollback_roundtrip_composite_primary_key():
    """Verify rollback roundtrip on tables with composite primary keys without column shifting or row loss."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE project_members (
            project_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            allocation_pct INTEGER NOT NULL,
            PRIMARY KEY (project_id, user_id)
        );
    """)
    conn.executemany(
        "INSERT INTO project_members VALUES (?, ?, ?, ?);",
        [
            ("PROJ_A", 1, "Lead", 50),
            ("PROJ_A", 2, "Dev", 100),
            ("PROJ_B", 1, "Dev", 50),
        ],
    )
    conn.commit()

    up_sql = "ALTER TABLE project_members ADD COLUMN notes TEXT DEFAULT '';"
    down_sql = "ALTER TABLE project_members DROP COLUMN notes;"

    res = verify_rollback_roundtrip(conn, up_sql, down_sql, target_table="project_members")
    assert res.classification == "EXACT"
    assert res.rollback_verified is True
    assert res.error is None
    assert len(res.lossy_columns) == 0


def test_rollback_roundtrip_no_primary_key_rowid():
    """Verify rollback roundtrip on tables without an explicit primary key using rowid."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE log_entries (
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL
        );
    """)
    conn.executemany(
        "INSERT INTO log_entries VALUES (?, ?, ?);",
        [
            ("2026-01-01T00:00:00", "INFO", "Server started"),
            ("2026-01-01T00:01:00", "WARN", "Memory high"),
        ],
    )
    conn.commit()

    up_sql = "ALTER TABLE log_entries ADD COLUMN host TEXT DEFAULT 'localhost';"
    down_sql = "ALTER TABLE log_entries DROP COLUMN host;"

    res = verify_rollback_roundtrip(conn, up_sql, down_sql, target_table="log_entries")
    assert res.classification == "EXACT"
    assert res.rollback_verified is True
    assert res.error is None


def test_parse_sql_scripts_varied_markdown_formats():
    """Verify parse_sql_scripts successfully extracts UP and DOWN scripts from markdown headings and sentences."""
    from timemachine.agent.loop import parse_sql_scripts

    c1 = """
### Migration
```sql
ALTER TABLE employees ADD COLUMN dept TEXT;
```
### Rollback Script
```sql
ALTER TABLE employees DROP COLUMN dept;
```
"""
    up1, down1 = parse_sql_scripts(c1)
    assert "ADD COLUMN dept" in up1
    assert "DROP COLUMN dept" in down1

    c2 = """
Here is the UP migration:
```sql
ALTER TABLE employees ADD COLUMN bonus INTEGER;
```
To rollback this migration:
```sql
ALTER TABLE employees DROP COLUMN bonus;
```
"""
    up2, down2 = parse_sql_scripts(c2)
    assert "ADD COLUMN bonus" in up2
    assert "DROP COLUMN bonus" in down2

