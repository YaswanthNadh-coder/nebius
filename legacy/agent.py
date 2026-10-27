"""Backwards-compatible API wrapper routing into the timemachine package."""

import sqlite3
import time
from collections.abc import Callable
from typing import Any

from faker import Faker
from openai import OpenAI

from timemachine.agent.llm import (
    call_llm as _tm_call_llm,
)
from timemachine.agent.llm import (
    clean_sql,
)
from timemachine.agent.llm import (
    get_client as _tm_get_client,
)
from timemachine.agent.llm import (
    resolve_nebius_models as _tm_resolve_nebius_models,
)
from timemachine.config import (
    AgentError,
    get_api_key_info,
    get_settings,
    is_placeholder_key,
    load_env_file,
)
from timemachine.verify.invariants import evaluate_migration_invariants
from timemachine.verify.sandbox import apply_sandbox

# Load env safely with UTF-8 BOM tolerance
load_env_file()

DEFAULT_FAST_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_ULTRA_MODEL = "nvidia/Nemotron-3-Ultra-550b-a55b"


def get_client(api_key_override: str | None = None) -> OpenAI:
    """Build an OpenAI-compatible client for Nebius Token Factory."""
    return _tm_get_client(api_key_override)


def resolve_nebius_models(client: OpenAI) -> tuple[str, str]:
    """Auto-detects Nemotron Super and Ultra model IDs available on the user's Nebius account."""
    fast, ultra, _ = _tm_resolve_nebius_models(client)
    return fast, ultra


def create_dirty_production_snapshot(
    schema_sql: str,
    num_rows: int = 1000,
) -> tuple[sqlite3.Connection, dict[str, int]]:
    """Creates an in-memory SQLite DB and seeds rows with realistic + adversarial production edge cases."""
    fake = Faker()
    Faker.seed(42)
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema_sql)
    cursor = conn.cursor()

    null_depts = 0
    legacy_depts = 0
    single_names = 0
    null_salaries = 0

    for i in range(1, num_rows + 1):
        if i % 14 == 0:
            full_name = "O'Reilly 🚀"
        elif i % 7 == 0:
            full_name = "Madonna"
            single_names += 1
        else:
            full_name = fake.name()

        if i % 5 == 0:
            dept_code = None
            null_depts += 1
        elif i % 9 == 0:
            dept_code = "LEGACY_99"
            legacy_depts += 1
        else:
            dept_code = fake.random_element(["ENG", "FIN", "OPS"])

        if i % 8 == 0:
            salary_raw = None
            null_salaries += 1
        else:
            salary_raw = f"${fake.random_int(45000, 145000)}"

        is_active = None if i % 6 == 0 else 1

        cursor.execute(
            "INSERT INTO employees (id, full_name, dept_code, salary_text, is_active) VALUES (?, ?, ?, ?, ?)",
            (i, full_name, dept_code, salary_raw, is_active),
        )
    conn.commit()

    stats = {
        "total_rows": num_rows,
        "null_dept_rows": null_depts,
        "unmapped_legacy_dept_rows": legacy_depts,
        "single_word_name_rows": single_names,
        "null_salary_rows": null_salaries,
    }
    return conn, stats


def verify_migration_on_clone(
    base_conn: sqlite3.Connection,
    migration_sql: str,
    target_table: str = "employees",
    required_cols: list[str] | None = None,
    non_null_cols: list[str] | None = None,
    expected_row_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Clones in-memory DB in <5ms, executes migration under sandbox authorizer, and verifies invariants."""
    t0 = time.perf_counter()
    clone_conn = sqlite3.connect(":memory:")
    base_conn.backup(clone_conn)
    apply_sandbox(clone_conn)

    b_cur = base_conn.cursor()
    b_cur.execute(f"SELECT COUNT(*) FROM \"{target_table}\"")
    before_count = b_cur.fetchone()[0]

    try:
        clone_conn.executescript(migration_sql)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Run full deterministic invariant suite
        inv = evaluate_migration_invariants(
            base_conn,
            clone_conn,
            target_table=target_table,
            required_cols=required_cols,
            non_null_cols=non_null_cols,
            expected_row_fn=expected_row_fn,
            elapsed_ms=elapsed_ms,
        )

        return {
            "passed": inv.passed,
            "status": inv.status,
            "before_count": inv.before_count,
            "after_count": inv.after_count,
            "elapsed_ms": inv.elapsed_ms,
            "error": inv.failed_checks[0] if inv.failed_checks else None,
            "failed_checks": inv.failed_checks,
            "passed_checks": inv.passed_checks,
            "sample_rows": inv.sample_rows,
        }
    except sqlite3.Error as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        err_msg = str(exc)
        if "authoriz" in err_msg.lower():
            err_text = f"SANDBOX VIOLATION: Execution denied by security authorizer: {err_msg}"
        else:
            err_text = f"SQLITE EXECUTION CRASH: {err_msg}"
        return {
            "passed": False,
            "status": "FAILED",
            "before_count": before_count,
            "after_count": None,
            "elapsed_ms": elapsed_ms,
            "error": err_text,
            "failed_checks": [err_text],
            "passed_checks": [],
            "sample_rows": [],
        }
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "passed": False,
            "status": "FAILED",
            "before_count": before_count,
            "after_count": None,
            "elapsed_ms": elapsed_ms,
            "error": f"EXECUTION ERROR: {str(exc)}",
            "failed_checks": [str(exc)],
            "passed_checks": [],
            "sample_rows": [],
        }
    finally:
        clone_conn.close()


def run_time_machine_agent(
    schema_sql: str,
    migration_goal: str,
    num_rows: int = 1000,
    max_retries: int = 3,
    api_key_override: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Legacy/compat agent loop: Draft (Super) -> Sandbox Invariant Audit -> Repair (Ultra).

    Fixes the off-by-one repair loop bug: every repair step is audited and recorded.
    """
    client = get_client(api_key_override)
    fast_model, ultra_model = resolve_nebius_models(client)

    base_conn, trap_stats = create_dirty_production_snapshot(schema_sql, num_rows=num_rows)
    history: list[dict[str, Any]] = []

    # Attempt 1: First-Pass Draft with Nemotron Super
    draft_prompt = f"""You are a SQLite database migration generator.
Current Schema:
{schema_sql}

Migration Goal:
{migration_goal}

Write a standard first-pass SQLite migration script (CREATE TABLE employees_new, INSERT INTO employees_new ... INNER JOIN departments d ON employees.dept_code = d.code, DROP TABLE employees, ALTER TABLE employees_new RENAME TO employees).
Return ONLY valid SQLite SQL inside a ```sql ``` code block."""

    resp_content, _ = _tm_call_llm(client, fast_model, draft_prompt, temperature=0.2)
    current_sql = clean_sql(resp_content)
    current_model = fast_model

    for attempt in range(1, max_retries + 1):
        result = verify_migration_on_clone(base_conn, current_sql, target_table="employees")
        history.append(
            {
                "attempt": attempt,
                "model": current_model,
                "sql": current_sql,
                "passed": result["passed"],
                "status": result.get("status", "FAILED"),
                "before_count": result["before_count"],
                "after_count": result["after_count"],
                "elapsed_ms": result["elapsed_ms"],
                "error": result["error"],
                "sample_rows": result["sample_rows"],
            }
        )

        if result["passed"]:
            break

        # If not passed and we have remaining attempts, repair with Ultra
        if attempt < max_retries:
            current_model = ultra_model
            heal_prompt = f"""A SQLite schema migration failed dry-run verification on a {num_rows}-row dirty production snapshot.
Adversarial Production Edge Cases Present in Snapshot:
- `dept_code` contains NULLs and unmapped 'LEGACY_99' codes not in `departments`.
- `full_name` contains single-word names with no space (e.g. 'Madonna') and apostrophes/Unicode.
- `salary_text` contains NULLs and '$' prefixed strings (e.g. '$85000').
- `is_active` contains NULLs.

Original Schema:
{schema_sql}

Migration Goal:
{migration_goal}

Failed SQL Script (Attempt #{attempt}):
```sql
{current_sql}
```

Sandbox Failure Report:
{result['error']}

Write a complete, corrected, lossless SQLite migration script that:
1. Rebuilds `employees` with columns: `id`, `first_name`, `last_name`, `dept_code`, `salary_num`, `is_active`.
2. Uses `LEFT JOIN departments d ON e.dept_code = d.code` and `COALESCE(d.code, 'UNASSIGNED')` so 0 rows are lost and `dept_code` is never NULL.
3. Splits `full_name` safely using `CASE WHEN INSTR(TRIM(e.full_name), ' ') > 0 THEN SUBSTR(TRIM(e.full_name), 1, INSTR(TRIM(e.full_name), ' ') - 1) ELSE TRIM(e.full_name) END` for `first_name` and `CASE WHEN INSTR(TRIM(e.full_name), ' ') > 0 THEN SUBSTR(TRIM(e.full_name), INSTR(TRIM(e.full_name), ' ') + 1) ELSE 'N/A' END` for `last_name`.
4. Converts `salary_text` safely via `CAST(REPLACE(COALESCE(e.salary_text, '0'), '$', '') AS INTEGER)`.
5. Uses `COALESCE(e.is_active, 0)` so `is_active` is never NULL.

Return ONLY valid SQLite SQL inside a ```sql ``` code block."""

            heal_resp, _ = _tm_call_llm(client, ultra_model, heal_prompt, temperature=0.1)
            current_sql = clean_sql(heal_resp)

    base_conn.close()
    return history, trap_stats
