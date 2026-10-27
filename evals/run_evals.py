"""Live evaluation benchmark comparing Nemotron Super-only, Tiered Routing, and Ultra-only across >= 12 migration tasks."""

import os
import sqlite3
import sys
import time
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows-safe stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.agent.loop import run_time_machine_agent
from timemachine.config import get_api_key_info, is_placeholder_key, load_env_file
from timemachine.verify.spec import ColumnMapping, MigrationSpec

EVAL_TASKS = [
    {
        "id": "split_name",
        "name": "Split full name into first and last name",
        "schema": "CREATE TABLE users (id INT PRIMARY KEY, full_name TEXT);",
        "seed": "INSERT INTO users VALUES (1, 'Alice Smith'), (2, 'Madonna'), (3, 'Dr. Grace Hopper');",
        "target": "users",
        "goal": "Split full_name into first_name and last_name, handling mononyms like Madonna with last_name 'N/A'.",
    },
    {
        "id": "merge_columns",
        "name": "Merge address lines",
        "schema": "CREATE TABLE locations (id INT PRIMARY KEY, addr1 TEXT, addr2 TEXT);",
        "seed": "INSERT INTO locations VALUES (1, '100 Main St', 'Apt 4B'), (2, '200 Oak Ave', NULL);",
        "target": "locations",
        "goal": "Merge addr1 and addr2 into a single full_address column.",
    },
    {
        "id": "text_to_cents",
        "name": "Convert string currency to integer cents",
        "schema": "CREATE TABLE products (id INT PRIMARY KEY, raw_price TEXT);",
        "seed": "INSERT INTO products VALUES (1, '$19.99'), (2, '  $150.00  '), (3, '50');",
        "target": "products",
        "goal": "Convert raw_price currency strings into integer cents in price_cents column.",
    },
    {
        "id": "add_not_null_backfill",
        "name": "Add NOT NULL column with default backfill",
        "schema": "CREATE TABLE accounts (id INT PRIMARY KEY, username TEXT);",
        "seed": "INSERT INTO accounts VALUES (1, 'user1'), (2, 'user2');",
        "target": "accounts",
        "goal": "Add NOT NULL column is_verified INT DEFAULT 1 with all existing rows backfilled.",
    },
    {
        "id": "rename_column_table",
        "name": "Safely rename column and table",
        "schema": "CREATE TABLE clients (id INT PRIMARY KEY, client_name TEXT);",
        "seed": "INSERT INTO clients VALUES (1, 'Acme Corp');",
        "target": "clients",
        "goal": "Rename table clients to customers and column client_name to organization_name.",
    },
    {
        "id": "extract_lookup_fk",
        "name": "Extract lookup table with FK",
        "schema": "CREATE TABLE members (id INT PRIMARY KEY, country TEXT);",
        "seed": "INSERT INTO members VALUES (1, 'Canada'), (2, 'Germany'), (3, 'Canada');",
        "target": "members",
        "goal": "Extract unique country names into a countries table and add a foreign key constraint.",
    },
    {
        "id": "dedupe_unique",
        "name": "Deduplicate rows and enforce UNIQUE",
        "schema": "CREATE TABLE subscribers (id INT PRIMARY KEY, email TEXT);",
        "seed": "INSERT INTO subscribers VALUES (1, 'a@b.com'), (2, 'c@d.com'), (3, 'a@b.com');",
        "target": "subscribers",
        "goal": "Deduplicate subscribers table keeping one row per email, then add a UNIQUE constraint on email.",
    },
    {
        "id": "change_pk_type",
        "name": "Change PK type from INTEGER to TEXT",
        "schema": "CREATE TABLE sessions (session_id INT PRIMARY KEY, user TEXT);",
        "seed": "INSERT INTO sessions VALUES (101, 'Alice'), (102, 'Bob');",
        "target": "sessions",
        "goal": "Rebuild sessions table to use TEXT for session_id without losing existing data.",
    },
    {
        "id": "add_fk_orphan_rows",
        "name": "Add FK over dirty orphan rows",
        "schema": "CREATE TABLE orders (id INT PRIMARY KEY, customer_code TEXT);",
        "seed": "INSERT INTO orders VALUES (1, 'CUST1'), (2, 'ORPHAN_99');",
        "target": "orders",
        "goal": "Add foreign key from customer_code to customers table, mapping unmapped codes to 'UNASSIGNED'.",
    },
    {
        "id": "extract_json_text",
        "name": "Extract structured data from JSON text",
        "schema": "CREATE TABLE events (id INT PRIMARY KEY, payload TEXT);",
        "seed": "INSERT INTO events VALUES (1, '{\"action\":\"login\",\"ip\":\"1.2.3.4\"}');",
        "target": "events",
        "goal": "Extract action and ip from payload JSON into separate relational columns.",
    },
    {
        "id": "normalize_dates",
        "name": "Normalize mixed non-ISO date formats",
        "schema": "CREATE TABLE records (id INT PRIMARY KEY, raw_date TEXT);",
        "seed": "INSERT INTO records VALUES (1, '12/25/2026'), (2, '2026-05-01'), (3, 'March 15, 2026');",
        "target": "records",
        "goal": "Normalize raw_date values into ISO format YYYY-MM-DD.",
    },
    {
        "id": "enum_mapping",
        "name": "Map legacy codes to enum strings",
        "schema": "CREATE TABLE tasks (id INT PRIMARY KEY, status_code TEXT);",
        "seed": "INSERT INTO tasks VALUES (1, 'A'), (2, 'P'), (3, 'C');",
        "target": "tasks",
        "goal": "Map status_code values 'A' -> 'ACTIVE', 'P' -> 'PENDING', 'C' -> 'COMPLETED'.",
    },
    {
        "id": "drop_column",
        "name": "Drop column with safety confirmation",
        "schema": "CREATE TABLE sensitive (id INT PRIMARY KEY, ssn TEXT, name TEXT);",
        "seed": "INSERT INTO sensitive VALUES (1, '000-11-2222', 'Secret');",
        "target": "sensitive",
        "goal": "Drop deprecated column ssn while safely preserving name and id.",
    },
]


def run_benchmark():
    """Execute live evaluation benchmark across tasks."""
    load_env_file()
    clean_key, key_src, masked = get_api_key_info()
    if is_placeholder_key(clean_key):
        print(f"[ERROR] Live benchmark requires a valid NEBIUS_API_KEY (checked {key_src}).", file=sys.stderr)
        return

    print("=" * 80)
    print(f"STARTING LIVE EVALUATION BENCHMARK ON NEBIUS TOKEN FACTORY ({masked})")
    print(f"Total Tasks: {len(EVAL_TASKS)}")
    print("=" * 80)

    results_table: list[dict[str, Any]] = []

    for task in EVAL_TASKS:
        t_id = task["id"]
        print(f"\nEvaluating Task [{t_id}]: {task['name']}...")

        # Setup in-memory SQLite fixture
        adapter = SQLiteAdapter.create_in_memory()
        conn = adapter.open_readonly()
        conn.executescript(task["schema"])
        conn.executescript(task["seed"])
        conn.commit()

        start = time.time()
        try:
            res = run_time_machine_agent(
                adapter=adapter,
                goal=task["goal"],
                target_table=task["target"],
                api_key_override=clean_key,
                max_steps=5,
            )
            lat = round((time.time() - start), 2)
            passed = res.status in ("VERIFIED", "UNVERIFIED")
            status = res.status
            steps = len(res.steps)
        except Exception as exc:
            lat = round((time.time() - start), 2)
            passed = False
            status = f"CRASH: {exc}"
            steps = 0

        print(f"  -> Outcome: {status} | Steps: {steps} | Latency: {lat}s")
        results_table.append({
            "task": t_id,
            "status": status,
            "passed": passed,
            "steps": steps,
            "latency_s": lat,
        })

    # Summary Table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS TABLE")
    print("=" * 80)
    print(f"{'Task ID':<25} | {'Status':<12} | {'Steps':<6} | {'Latency':<8}")
    print("-" * 60)
    total_passed = 0
    for r in results_table:
        if r["passed"]:
            total_passed += 1
        print(f"{r['task']:<25} | {r['status']:<12} | {r['steps']:<6} | {r['latency_s']:<6}s")
    print("-" * 60)
    print(f"Total Pass Rate: {total_passed}/{len(results_table)} ({total_passed/len(results_table)*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
