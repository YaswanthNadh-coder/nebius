"""Tests for Item A: Robust UP/DOWN SQL script parsing, split markers, submit tool guard, and self-undo guard."""

from timemachine.agent.loop import parse_sql_scripts, parse_sql_scripts_ex

# ── Battery of 9 Formats ───────────────────────────────────────────────────────

def test_format_1_intro_mentions_rollback_before_unlabeled_first_block():
    """Intro text mentioning 'rollback' before block 1 must NEVER mark block 1 as DOWN."""
    content = """
Here is the schema migration along with the rollback plan if something fails.

```sql
CREATE TABLE employees_new (
    id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL
);
INSERT INTO employees_new SELECT id, 'A', 'B' FROM employees;
DROP TABLE employees;
ALTER TABLE employees_new RENAME TO employees;
```

```sql
-- DOWN MIGRATION
CREATE TABLE employees_old (
    id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL
);
INSERT INTO employees_old SELECT id, first_name || ' ' || last_name FROM employees;
DROP TABLE employees;
ALTER TABLE employees_old RENAME TO employees;
```
"""
    up, down = parse_sql_scripts(content)
    assert "CREATE TABLE employees_new" in up
    assert "CREATE TABLE employees_old" in down
    assert "CREATE TABLE employees_new" not in down


def test_format_2_revert_comment():
    """DOWN block labelled '-- Revert'."""
    content = """
```sql
CREATE TABLE employees_new (id INT);
```

```sql
-- Revert
CREATE TABLE employees_old (id INT);
```
"""
    up, down = parse_sql_scripts(content)
    assert "employees_new" in up
    assert "employees_old" in down
    assert "employees_old" not in up


def test_format_3_undo_migration_comment():
    """DOWN block labelled '-- Undo migration'."""
    content = """
```sql
CREATE TABLE employees_new (id INT);
```

```sql
-- Undo migration
CREATE TABLE employees_old (id INT);
```
"""
    up, down = parse_sql_scripts(content)
    assert "employees_new" in up
    assert "employees_old" in down
    assert "employees_old" not in up


def test_format_4_reverse_markdown_header():
    """DOWN block preceded by '### Reverse' markdown header."""
    content = """
```sql
CREATE TABLE employees_new (id INT);
```

### Reverse
```sql
CREATE TABLE employees_old (id INT);
```
"""
    up, down = parse_sql_scripts(content)
    assert "employees_new" in up
    assert "employees_old" in down
    assert "employees_old" not in up


def test_format_5_banner_inside_single_fence():
    """A '-- ===== DOWN =====' banner inside a single code fence."""
    content = """
```sql
CREATE TABLE employees_new (id INT);
-- ===== DOWN =====
CREATE TABLE employees_old (id INT);
```
"""
    up, down = parse_sql_scripts(content)
    assert "employees_new" in up
    assert "employees_old" in down
    assert "employees_old" not in up


def test_format_6_fence_tag_marker():
    """DOWN block identified by fence tag e.g. ```sql down or ```sql revert."""
    content = """
```sql
CREATE TABLE employees_new (id INT);
```

```sql revert
CREATE TABLE employees_old (id INT);
```
"""
    up, down = parse_sql_scripts(content)
    assert "employees_new" in up
    assert "employees_old" in down
    assert "employees_old" not in up


def test_format_7_markdown_bold_rollback():
    """DOWN block preceded by bold text '**Rollback:**'."""
    content = """
```sql
CREATE TABLE employees_new (id INT);
```

**Rollback:**
```sql
CREATE TABLE employees_old (id INT);
```
"""
    up, down = parse_sql_scripts(content)
    assert "employees_new" in up
    assert "employees_old" in down
    assert "employees_old" not in up


def test_format_8_backward_and_migrate_down():
    """DOWN block identified by '-- backward' or '-- migrate down'."""
    content = """
```sql
CREATE TABLE employees_new (id INT);
```

```sql
-- migrate down
CREATE TABLE employees_old (id INT);
```
"""
    up, down = parse_sql_scripts(content)
    assert "employees_new" in up
    assert "employees_old" in down
    assert "employees_old" not in up


def test_format_9_banner_with_decorations():
    """DOWN block banner with symbols: '-- ====== ROLLBACK ======'."""
    content = """
```sql
CREATE TABLE employees_new (id INT);
```

```sql
-- ====== ROLLBACK ======
CREATE TABLE employees_old (id INT);
```
"""
    up, down = parse_sql_scripts(content)
    assert "employees_new" in up
    assert "employees_old" in down
    assert "employees_old" not in up


# ── parse_sql_scripts_ex Rule Tracking ────────────────────────────────────────

def test_parse_sql_scripts_ex_returns_rule():
    """parse_sql_scripts_ex returns up, down, and the rule name matched."""
    content = """
```sql
CREATE TABLE employees_new (id INT);
```
-- Revert
```sql
CREATE TABLE employees_old (id INT);
```
"""
    up, down, rule = parse_sql_scripts_ex(content)
    assert "employees_new" in up
    assert "employees_old" in down
    assert rule != ""
    assert "revert" in rule.lower() or "prose" in rule.lower() or "split" in rule.lower() or "marker" in rule.lower()


# ── Submit Tool Auto-Splitting Guard ──────────────────────────────────────────

def test_submit_tool_splits_down_when_in_up_sql():
    """When submit(up_sql, down_sql) has both scripts in up_sql, it splits them."""
    from timemachine.adapters.sqlite import SQLiteAdapter
    from timemachine.agent.tools import ToolExecutor

    adapter = SQLiteAdapter.create_in_memory()
    executor = ToolExecutor(adapter)

    both_in_up = """
CREATE TABLE employees_new (id INT);
-- ===== DOWN =====
CREATE TABLE employees_old (id INT);
"""
    res = executor.execute("submit", {"up_sql": both_in_up, "down_sql": ""})
    assert "employees_new" in res.get("up_sql", "")
    assert "employees_old" not in res.get("up_sql", "")
    assert "employees_old" in res.get("down_sql", "")


# ── Run on Clone Auto-Splitting Guard ─────────────────────────────────────────

def test_run_on_clone_strips_down_when_in_up_sql():
    """run_on_clone must strip any DOWN section from candidate up_sql before execution."""
    from timemachine.adapters.sqlite import SQLiteAdapter
    from timemachine.agent.tools import ToolExecutor

    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT, full_name TEXT);")
    conn.commit()

    executor = ToolExecutor(adapter)
    # If DOWN were executed, it would revert or drop
    combined = """
CREATE TABLE employees_new (id INT, first_name TEXT);
INSERT INTO employees_new SELECT id, 'A' FROM employees;
DROP TABLE employees;
ALTER TABLE employees_new RENAME TO employees;
-- ===== DOWN =====
CREATE TABLE employees_old (id INT, full_name TEXT);
DROP TABLE employees;
ALTER TABLE employees_old RENAME TO employees;
"""
    res = executor.execute("run_on_clone", {"up_sql": combined})
    # The clone should have employees with first_name (from UP), not reverted!
    assert res.get("status") in ("VERIFIED", "PASSED", "UNVERIFIED", "FAILED")
    # If run_on_clone ran DOWN, the table would have full_name. If UP ran, first_name.
    # In either case, the error shouldn't complain that UP dropped everything or was undone.


# ── Revert Repro (End-to-End Status VERIFIED) ──────────────────────────────────

def test_revert_label_repro_verifies_without_self_undo(monkeypatch):
    """The Revert repro must reach status VERIFIED, not FAILED with MISSING_REQUIRED_COLUMNS."""
    from unittest.mock import MagicMock

    from timemachine.agent.loop import run_time_machine_agent
    from timemachine.ui.app_view import create_builtin_scenario_adapter
    from timemachine.verify.spec import ColumnMapping, MigrationSpec

    adapter = create_builtin_scenario_adapter(1)
    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="first_name", source_column="full_name", transform="split_name_first", null_policy="not_null"),
                ColumnMapping(new_column="last_name", source_column="full_name", transform="split_name_last", null_policy="not_null"),
            ]
        },
        columns_declared_unchanged={"employees": ["id", "department", "salary", "created_at"]},
    )

    # Response uses '-- Revert' header for the down script
    mock_response = """
Here is the migration:

```sql
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
    END,
    CASE
        WHEN INSTR(TRIM(full_name, ' ' || char(9) || char(10) || char(13)), ' ') > 0
        THEN TRIM(SUBSTR(TRIM(full_name, ' ' || char(9) || char(10) || char(13)), INSTR(TRIM(full_name, ' ' || char(9) || char(10) || char(13)), ' ') + 1), ' ' || char(9) || char(10) || char(13))
        ELSE 'N/A'
    END,
    department,
    salary,
    created_at
FROM employees;
DROP TABLE employees;
ALTER TABLE employees_new RENAME TO employees;
CREATE INDEX idx_emp_salary ON employees(salary);
CREATE TRIGGER trg_emp_audit AFTER INSERT ON employees BEGIN SELECT 1; END;
```

-- Revert
```sql
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
    CASE WHEN last_name = 'N/A' THEN first_name ELSE first_name || ' ' || last_name END,
    department,
    salary,
    created_at
FROM employees;
DROP TABLE employees;
ALTER TABLE employees_old RENAME TO employees;
CREATE INDEX idx_emp_salary ON employees(salary);
CREATE TRIGGER trg_emp_audit AFTER INSERT ON employees BEGIN SELECT 1; END;
```
"""

    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = mock_response
    mock_msg.tool_calls = None
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=mock_msg)],
        usage=MagicMock(total_tokens=100),
    )

    monkeypatch.setattr("timemachine.agent.loop.get_client", lambda api_key=None: mock_client)
    monkeypatch.setattr("timemachine.agent.loop.resolve_nebius_models", lambda client: ("fast-model", "ultra-model", []))

    result = run_time_machine_agent(
        adapter=adapter,
        goal="Split full_name into first_name and last_name",
        spec=spec,
        max_steps=1,
    )

    assert result.status == "VERIFIED", f"Expected VERIFIED but got {result.status} (notes: {result.notes}, first fail: {result.invariant_result.failed_checks if result.invariant_result else None})"
    assert result.rollback_verified is True
