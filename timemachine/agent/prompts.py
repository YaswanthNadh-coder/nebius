"""Prompt templates for planning, drafting, and repairing schema migrations."""

import json

from timemachine.adapters.schema import DatabaseSchema
from timemachine.verify.invariants import InvariantResult
from timemachine.verify.spec import MigrationSpec


def format_schema_summary(schema: DatabaseSchema) -> str:
    """Format schema into concise text description for LLM prompts."""
    lines: list[str] = []
    for tname, tbl in schema.tables.items():
        cols_desc = []
        for c in tbl.columns:
            pk_tag = " PRIMARY KEY" if c.pk else ""
            nn_tag = " NOT NULL" if c.notnull else ""
            def_tag = f" DEFAULT {c.default}" if c.default is not None else ""
            cols_desc.append(f"  {c.name} {c.type}{pk_tag}{nn_tag}{def_tag}")

        lines.append(f"Table {tname} ({tbl.row_count} rows):")
        lines.extend(cols_desc)

        if tbl.foreign_keys:
            for fk in tbl.foreign_keys:
                lines.append(f"  FK: {fk.from_col} -> {fk.to_table}({fk.to_col})")
        if tbl.indexes:
            for idx in tbl.indexes:
                u = "UNIQUE " if idx.unique else ""
                lines.append(f"  INDEX {u}{idx.name} on ({', '.join(idx.columns)})")

    if schema.views:
        lines.append("\nViews:")
        for vname, v in schema.views.items():
            lines.append(f"  VIEW {vname}: {v.sql}")

    if schema.triggers:
        lines.append("\nTriggers:")
        for trname, tr in schema.triggers.items():
            lines.append(f"  TRIGGER {trname} ON {tr.table}: {tr.sql}")

    return "\n".join(lines)


def build_plan_prompt(schema: DatabaseSchema, goal: str, profiles: dict | None = None) -> str:
    """Build prompt asking model to generate a structured MigrationSpec JSON."""
    schema_text = format_schema_summary(schema)
    profile_summary = ""
    if profiles:
        profile_summary = f"\nData Profile Summary (Masked):\n{json.dumps(profiles, indent=2, default=str)}"

    return f"""You are an expert database migration architect.
Your task is to analyze the database schema and migration goal, then output a structured MigrationSpec in JSON.

Database Schema:
{schema_text}
{profile_summary}

Migration Goal:
{goal}

Rules for MigrationSpec JSON:
1. `affected_tables`: List of tables modified or created.
2. `column_mappings`: For each affected table, list of objects:
   - `new_column`: name of target column.
   - `source_column`: original column name (or null if computed).
   - `transform`: one of ["passthrough", "split_name_first", "split_name_last", "strip_honorific", "cast_int", "cast_cents", "default_if_null", "map_via_lookup", "normalize_date", "salary_band", "custom"].
   - `null_policy`: "not_null" or "nullable".
   - `default_value`: fallback default value if null.
   - `lookup_table`: lookup table name if transform is map_via_lookup.
   - `lookup_key`: lookup code column.
   - `lookup_val`: lookup target column.
3. `row_policy`: "preserve_all" (zero row loss) or "deletions_allowed".
4. `columns_declared_unchanged`: mapping of table name to columns whose row data must not be touched.
5. `expected_constraints`: table constraints to enforce (e.g. PRIMARY KEY, UNIQUE, NOT NULL).
6. `risky_operations`: any destructive operations needed (e.g. DROP TABLE old).

Return ONLY the JSON object inside a ```json ``` block."""


def build_migration_prompt(
    schema: DatabaseSchema,
    spec: MigrationSpec,
    is_naive_baseline: bool = False,
) -> str:
    """Build prompt for generating up.sql and down.sql."""
    schema_text = format_schema_summary(schema)
    spec_json = spec.model_dump_json(indent=2)

    instruction = (
        "Write a complete, safe, production-grade SQLite migration script.\n"
        "Requirements:\n"
        "1. Rebuild tables following SQLite standard 12-step table rebuild (create new table, copy data, drop old, rename, recreate indexes/triggers/views).\n"
        "2. Zero silent row drops: Use LEFT JOIN and COALESCE so dirty foreign keys or unmapped codes default safely without losing records.\n"
        "3. Output BOTH ```sql -- UP MIGRATION ``` and ```sql -- DOWN MIGRATION (ROLLBACK) ```.\n"
        "4. You can use available tools (get_schema, profile_table, find_violating_rows, run_on_clone) to inspect data or test candidate fixes on the disposable sandbox clone.\n"
        "5. When complete, call the submit tool or provide the migration scripts inside ```sql blocks."
    )
    if is_naive_baseline:
        instruction = (
            "Write a standard first-pass naive migration script using INNER JOIN (as a baseline test).\n"
            "Output the migration script inside a ```sql ``` block."
        )

    return f"""You are a database migration engineer writing SQLite migration SQL.

Current Database Schema:
{schema_text}

Approved Migration Plan (MigrationSpec):
{spec_json}

{instruction}
"""


def build_repair_prompt(
    schema: DatabaseSchema,
    spec: MigrationSpec,
    failed_sql: str,
    result: InvariantResult | None = None,
    trap_results: list[dict] | None = None,
) -> str:
    """Build repair prompt with structured failure telemetry."""
    schema_text = format_schema_summary(schema)
    if result is None:
        truncated_reply = (failed_sql or "")[:500]
        return f"""You are a database migration engineer writing SQLite migration SQL.

Target Schema:
{schema_text}
your previous reply contained no executable SQL (reply was: {truncated_reply}); reply with the migration.

Output the complete, corrected SQLite migration inside a ```sql ``` block with both UP and DOWN migrations, or call submit."""

    failed_checks_text = "\n".join([f"- {fc}" for fc in result.failed_checks])
    traps_text = ""
    if trap_results:
        failing_traps = [t for t in trap_results if not t.get("survived")]
        if failing_traps:
            traps_text = "\nFailed Adversarial Traps:\n" + "\n".join(
                [f"- Trap {t.get('trap_type')}: {t.get('description')} (pk={t.get('pk')})" for t in failing_traps]
            )

    return f"""The previous SQLite schema migration failed dry-run verification on the sandbox clone.

Target Schema:
{schema_text}

Failed SQL Script:
```sql
{failed_sql}
```

Sandbox Failure Telemetry:
Row Counts: Before={result.before_count}, After={result.after_count}
Failed Checks:
{failed_checks_text}
{traps_text}

Task:
Analyze the exact failure points above and write a complete, corrected, lossless SQLite migration.
1. Fix row loss or duplication (use LEFT JOIN and COALESCE for unmatched foreign keys).
2. Fix any missing columns or violated NOT NULL constraints.
3. Recreate all pre-existing indexes, views, and triggers.
4. Provide both UP migration and DOWN rollback migration.

You may use available tools (get_schema, profile_table, find_violating_rows, run_on_clone) to inspect the data or test candidate fixes on the sandbox clone.
When complete, call the submit tool or return the corrected scripts in ```sql blocks with both UP and DOWN migrations."""
