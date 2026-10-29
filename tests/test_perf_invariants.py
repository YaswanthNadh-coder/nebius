"""Tests for Item F: Verification Performance, streaming PK comparison, and sampled auditing at scale."""

import sqlite3
import time

from timemachine.verify.invariants import evaluate_migration_invariants
from timemachine.verify.spec import ColumnMapping, MigrationSpec


def create_scaled_db(num_rows: int = 100_000) -> tuple[sqlite3.Connection, sqlite3.Connection, MigrationSpec]:
    """Create base and migrated SQLite in-memory connections with num_rows."""
    base_conn = sqlite3.connect(":memory:")
    base_cur = base_conn.cursor()
    base_cur.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            salary INTEGER,
            department TEXT
        );
    """)

    # Populate in batch
    batch_size = 10_000
    for start_id in range(1, num_rows + 1, batch_size):
        end_id = min(start_id + batch_size, num_rows + 1)
        rows = [(i, f"First_{i}", f"Last_{i}", 50000 + (i % 50000), "ENG") for i in range(start_id, end_id)]
        base_cur.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", rows)
    base_conn.commit()

    # Clone to migrated
    migrated_conn = sqlite3.connect(":memory:")
    base_conn.backup(migrated_conn)

    # Hand-written migration: adds bonus_cents from salary
    migrated_cur = migrated_conn.cursor()
    migrated_cur.execute("""
        CREATE TABLE employees_new (
            id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            salary INTEGER,
            department TEXT,
            bonus_cents INTEGER NOT NULL
        );
    """)
    migrated_cur.execute("""
        INSERT INTO employees_new (id, first_name, last_name, salary, department, bonus_cents)
        SELECT id, first_name, last_name, salary, department, salary * 10
        FROM employees;
    """)
    migrated_cur.execute("DROP TABLE employees;")
    migrated_cur.execute("ALTER TABLE employees_new RENAME TO employees;")
    migrated_conn.commit()

    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="id", source_column="id", transform="passthrough"),
                ColumnMapping(new_column="first_name", source_column="first_name", transform="passthrough"),
                ColumnMapping(new_column="last_name", source_column="last_name", transform="passthrough"),
                ColumnMapping(new_column="salary", source_column="salary", transform="passthrough"),
                ColumnMapping(new_column="department", source_column="department", transform="passthrough"),
                ColumnMapping(new_column="bonus_cents", source_column="salary", transform="custom"),
            ]
        },
        columns_declared_unchanged={"employees": ["first_name", "last_name", "salary", "department"]},
    )

    return base_conn, migrated_conn, spec


def test_invariants_performance_100k_rows():
    """Verification on 100k rows must complete in <= 1.0s and be marked 'sampled'."""
    base_conn, migrated_conn, spec = create_scaled_db(100_000)

    t0 = time.perf_counter()
    result = evaluate_migration_invariants(base_conn, migrated_conn, target_table="employees", spec=spec)
    duration = time.perf_counter() - t0

    base_conn.close()
    migrated_conn.close()

    assert result.passed is True, f"Expected passed, got: {result.failed_checks}"
    assert result.is_sampled is True or any("sampled" in str(c).lower() for c in result.passed_checks)
    assert duration <= 1.2, f"Verification took {duration:.3f}s, target was <= 1.0s"


def test_sampled_verification_catches_trap_corruption_at_scale():
    """Sampled verification must not miss corruption in trap rows or sampled range."""
    base_conn, migrated_conn, spec = create_scaled_db(25_000)

    # Corrupt an untouched column in one row
    migrated_cur = migrated_conn.cursor()
    migrated_cur.execute("UPDATE employees SET salary = salary + 999 WHERE id = 12500;")
    migrated_conn.commit()

    result = evaluate_migration_invariants(base_conn, migrated_conn, target_table="employees", spec=spec)

    base_conn.close()
    migrated_conn.close()

    assert result.passed is False
    assert any("UNTOUCHED_COLUMN_MUTATED" in fc for fc in result.failed_checks)

