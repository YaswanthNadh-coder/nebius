import os
import re
import time
import sqlite3
from dotenv import load_dotenv
from faker import Faker
from openai import OpenAI

load_dotenv()

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# Default Nebius Token Factory Model IDs (with automatic fallback detection)
DEFAULT_FAST_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_ULTRA_MODEL = "nvidia/Nemotron-3-Ultra-550b-a55b"


def get_client(api_key_override: str | None = None) -> OpenAI:
    key = api_key_override or os.environ.get("NEBIUS_API_KEY", "")
    return OpenAI(base_url=NEBIUS_BASE_URL, api_key=key)


def resolve_nebius_models(client: OpenAI) -> tuple[str, str]:
    """Auto-detects exact Nemotron Super/Nano and Ultra model IDs available on the user's Nebius account."""
    try:
        models = [m.id for m in client.models.list().data]
        nemotrons = [m for m in models if "nemotron" in m.lower() or "nvidia" in m.lower()]
        if not nemotrons:
            return DEFAULT_FAST_MODEL, DEFAULT_ULTRA_MODEL

        fast_candidates = [m for m in nemotrons if "super" in m.lower() or "nano" in m.lower() or "lightning" in m.lower()]
        ultra_candidates = [m for m in nemotrons if "ultra" in m.lower()]

        fast_model = fast_candidates[0] if fast_candidates else nemotrons[0]
        ultra_model = ultra_candidates[0] if ultra_candidates else fast_model
        return fast_model, ultra_model
    except Exception:
        return DEFAULT_FAST_MODEL, DEFAULT_ULTRA_MODEL


def clean_sql(raw_text: str) -> str:
    """Strips <think> reasoning tags and markdown code fences to return pure executable SQLite SQL."""
    if not raw_text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def create_dirty_production_snapshot(schema_sql: str, num_rows: int = 1000) -> tuple[sqlite3.Connection, dict]:
    """Creates an in-memory SQLite DB and seeds 1,000 rows with realistic + adversarial production edge cases."""
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
) -> dict:
    """Clones the in-memory DB in <5ms, executes the migration SQL, and audits for silent data loss."""
    required_cols = required_cols or ["id", "first_name", "last_name", "dept_code", "salary_num", "is_active"]
    non_null_cols = non_null_cols or ["first_name", "last_name", "dept_code", "salary_num", "is_active"]

    t0 = time.perf_counter()
    clone_conn = sqlite3.connect(":memory:")
    base_conn.backup(clone_conn)
    cur = clone_conn.cursor()

    cur.execute(f"SELECT COUNT(*) FROM {target_table}")
    before_count = cur.fetchone()[0]

    try:
        clone_conn.executescript(migration_sql)
        cur.execute(f"SELECT COUNT(*) FROM {target_table}")
        after_count = cur.fetchone()[0]
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Audit 1: Did a naive INNER JOIN or WHERE filter silently delete rows?
        if after_count != before_count:
            return {
                "passed": False,
                "before_count": before_count,
                "after_count": after_count,
                "elapsed_ms": elapsed_ms,
                "error": (
                    f"SILENT DATA LOSS DETECTED: Row count dropped from {before_count} to {after_count} "
                    f"({before_count - after_count} rows permanently lost!). Caused by filtering or INNER JOIN on NULL/unmapped dept_code."
                ),
                "sample_rows": [],
            }

        cur.execute(f"SELECT * FROM {target_table} LIMIT 8")
        cols = [description[0] for description in cur.description]
        sample = [dict(zip(cols, row)) for row in cur.fetchall()]

        # Audit 2: Are all required target columns present?
        for req in required_cols:
            if req not in cols:
                return {
                    "passed": False,
                    "before_count": before_count,
                    "after_count": after_count,
                    "elapsed_ms": elapsed_ms,
                    "error": f"SCHEMA MISMATCH: Required column '{req}' is missing from migrated table '{target_table}'. Found columns: {cols}",
                    "sample_rows": sample,
                }

        # Audit 3: Did string splitting or type casting introduce NULL corruption?
        for col in non_null_cols:
            cur.execute(f"SELECT COUNT(*) FROM {target_table} WHERE {col} IS NULL")
            null_cnt = cur.fetchone()[0]
            if null_cnt > 0:
                return {
                    "passed": False,
                    "before_count": before_count,
                    "after_count": after_count,
                    "elapsed_ms": elapsed_ms,
                    "error": f"NULL CORRUPTION DETECTED: Column '{col}' contains {null_cnt} unhandled NULL values after migration.",
                    "sample_rows": sample,
                }

        return {
            "passed": True,
            "before_count": before_count,
            "after_count": after_count,
            "elapsed_ms": elapsed_ms,
            "error": None,
            "sample_rows": sample,
        }

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "passed": False,
            "before_count": before_count,
            "after_count": 0,
            "elapsed_ms": elapsed_ms,
            "error": f"SQLITE EXECUTION CRASH: {str(exc)}",
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
) -> tuple[list[dict], dict]:
    """Main Agentic Loop: Fast Draft (Super) -> Sandbox Test -> Self-Heal (Ultra)."""
    client = get_client(api_key_override)
    fast_model, ultra_model = resolve_nebius_models(client)

    base_conn, trap_stats = create_dirty_production_snapshot(schema_sql, num_rows=num_rows)
    history = []

    # Step 1: Fast First-Pass Draft with Nemotron Super
    draft_prompt = f"""You are a SQLite database migration generator.
Current Schema:
{schema_sql}

Migration Goal:
{migration_goal}

Write a standard first-pass SQLite migration script (CREATE TABLE employees_new, INSERT INTO employees_new ... INNER JOIN departments d ON employees.dept_code = d.code, DROP TABLE employees, ALTER TABLE employees_new RENAME TO employees).
Return ONLY valid SQLite SQL inside a ```sql ``` code block."""

    resp = client.chat.completions.create(
        model=fast_model,
        messages=[{"role": "user", "content": draft_prompt}],
        temperature=0.2,
    )
    current_sql = clean_sql(resp.choices[0].message.content)
    current_model = fast_model

    for attempt in range(1, max_retries + 1):
        result = verify_migration_on_clone(base_conn, current_sql, target_table="employees")
        history.append(
            {
                "attempt": attempt,
                "model": current_model,
                "sql": current_sql,
                "passed": result["passed"],
                "before_count": result["before_count"],
                "after_count": result["after_count"],
                "elapsed_ms": result["elapsed_ms"],
                "error": result["error"],
                "sample_rows": result["sample_rows"],
            }
        )

        if result["passed"]:
            break

        # Step 2: Escalate to Nemotron Ultra for Deep Self-Healing
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

        heal_resp = client.chat.completions.create(
            model=ultra_model,
            messages=[{"role": "user", "content": heal_prompt}],
            temperature=0.1,
        )
        current_sql = clean_sql(heal_resp.choices[0].message.content)

    base_conn.close()
    return history, trap_stats
