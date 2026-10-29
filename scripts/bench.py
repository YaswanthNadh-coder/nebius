#!/usr/bin/env python3
"""Benchmark script measuring clone, profile, migration exec, verification, and rollback timings across scale tiers."""

import sqlite3
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.traps.profiler import profile_database
from timemachine.verify.invariants import evaluate_migration_invariants
from timemachine.verify.rebuild import verify_rollback_roundtrip
from timemachine.verify.spec import ColumnMapping, MigrationSpec


def create_benchmark_adapter(num_rows: int) -> tuple[SQLiteAdapter, MigrationSpec, str, str]:
    """Create in-memory SQLiteAdapter pre-populated with num_rows and return reference UP/DOWN and spec."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            salary INTEGER,
            department TEXT
        );
    """)

    batch_size = 20_000
    for start_id in range(1, num_rows + 1, batch_size):
        end_id = min(start_id + batch_size, num_rows + 1)
        rows = [
            (i, f"First_{i}", f"Last_{i}", 50000 + (i % 50000), "ENG")
            for i in range(start_id, end_id)
        ]
        cur.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()

    adapter = SQLiteAdapter.from_connection(conn)

    up_sql = """
    CREATE TABLE employees_new (
        id INTEGER PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        salary INTEGER,
        department TEXT,
        bonus_cents INTEGER NOT NULL
    );
    INSERT INTO employees_new (id, first_name, last_name, salary, department, bonus_cents)
    SELECT id, first_name, last_name, salary, department, salary * 10
    FROM employees;
    DROP TABLE employees;
    ALTER TABLE employees_new RENAME TO employees;
    """

    down_sql = """
    CREATE TABLE employees_old (
        id INTEGER PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        salary INTEGER,
        department TEXT
    );
    INSERT INTO employees_old (id, first_name, last_name, salary, department)
    SELECT id, first_name, last_name, salary, department
    FROM employees;
    DROP TABLE employees;
    ALTER TABLE employees_old RENAME TO employees;
    """

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

    return adapter, spec, up_sql, down_sql


def run_benchmarks():
    """Run benchmark across 10k, 100k, and 400k scale tiers."""
    tiers = [10_000, 100_000, 400_000]
    results = []

    print("=" * 80)
    print("  Time Machine Dry-Run Verification & Performance Benchmark")
    print("=" * 80)

    for num_rows in tiers:
        print(f"\n[+] Initializing scale tier: {num_rows:,} rows...")
        adapter, spec, up_sql, down_sql = create_benchmark_adapter(num_rows)

        # 1. Clone timing
        t0 = time.perf_counter()
        clone_conn = adapter.clone()
        clone_ms = (time.perf_counter() - t0) * 1000

        # 2. Profile timing
        schema = adapter.introspect()
        t0 = time.perf_counter()
        _ = profile_database(clone_conn, schema)
        profile_ms = (time.perf_counter() - t0) * 1000

        # 3. Migration exec timing
        test_conn = adapter.clone()
        ok, err, exec_ms = adapter.run_in_sandbox(test_conn, up_sql, fk_enabled=True)
        assert ok is True, f"Migration failed: {err}"

        # 4. Invariant & Oracle verification timing
        base_conn = adapter.clone()
        t0 = time.perf_counter()
        inv_res = evaluate_migration_invariants(base_conn, test_conn, target_table="employees", spec=spec)
        verify_ms = (time.perf_counter() - t0) * 1000
        assert inv_res.passed is True, f"Invariants failed: {inv_res.failed_checks}"

        # 5. Rollback roundtrip timing
        rb_conn = adapter.clone()
        t0 = time.perf_counter()
        rb_res = verify_rollback_roundtrip(rb_conn, up_sql, down_sql, target_table="employees")
        rollback_ms = (time.perf_counter() - t0) * 1000
        assert rb_res.rollback_verified is True, f"Rollback failed: {rb_res.error}"

        # Cleanup connections
        clone_conn.close()
        test_conn.close()
        base_conn.close()
        rb_conn.close()
        adapter.close()

        results.append({
            "rows": num_rows,
            "clone_ms": round(clone_ms, 2),
            "profile_ms": round(profile_ms, 2),
            "exec_ms": round(exec_ms, 2),
            "verify_ms": round(verify_ms, 2),
            "rollback_ms": round(rollback_ms, 2),
            "sampled": inv_res.is_sampled,
        })

    print("\n" + "=" * 80)
    print("  Benchmark Summary Table")
    print("=" * 80)
    print("| Scale Tier | Clone (ms) | Profile (ms) | Migration Exec (ms) | Invariants & Oracle (ms) | Rollback (ms) | Mode |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        mode_str = "Sampled (5k+traps)" if r["sampled"] else "Exact (Full)"
        print(f"| **{r['rows']:,} rows** | {r['clone_ms']} ms | {r['profile_ms']} ms | {r['exec_ms']} ms | {r['verify_ms']} ms | {r['rollback_ms']} ms | {mode_str} |")

    print("\nTargets Checklist:")
    for r in results:
        if r["rows"] == 100_000:
            status = "PASS" if r["verify_ms"] <= 1000 else "FAIL"
            print(f"- 100k rows verification target <= 1,000ms: {r['verify_ms']}ms [{status}]")
        elif r["rows"] == 400_000:
            status = "PASS" if r["verify_ms"] <= 3000 else "FAIL"
            print(f"- 400k rows verification target <= 3,000ms: {r['verify_ms']}ms [{status}]")


if __name__ == "__main__":
    run_benchmarks()
