"""Prompt templates for drafting and repairing schema migrations with transform contracts and gotchas."""

from typing import Any

from timemachine.adapters.schema import DatabaseSchema
from timemachine.verify import oracle
from timemachine.verify.invariants import InvariantResult
from timemachine.verify.sandbox import ALLOWED_PRAGMAS
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


def generate_transform_contract(spec: MigrationSpec) -> str:
    """Generate transform contract from the spec using exact docstrings from oracle.py."""
    if not spec.column_mappings:
        return ""

    lines = ["Transform Contract (exact reference oracle verification rules):"]
    for tname, mappings in spec.column_mappings.items():
        lines.append(f"Table {tname}:")
        for m in mappings:
            trans_name = (m.transform or "passthrough").lower()
            # Map aliases to oracle function names
            func_name = trans_name
            if trans_name in ("first_name",):
                func_name = "split_name_first"
            elif trans_name in ("last_name",):
                func_name = "split_name_last"
            elif trans_name in ("clean_name",):
                func_name = "strip_honorific"
            elif trans_name in ("integer",):
                func_name = "cast_int"
            elif trans_name in ("cents",):
                func_name = "cast_cents"
            elif trans_name in ("coalesce",):
                func_name = "default_if_null"
            elif trans_name in ("lookup",):
                func_name = "map_via_lookup"
            elif trans_name in ("date",):
                func_name = "normalize_date"
            elif trans_name in ("identity",):
                func_name = "passthrough"

            fn = getattr(oracle, func_name, None)
            doc = fn.__doc__.strip() if fn and fn.__doc__ else f"Apply {trans_name} transform."
            src_str = f" from {m.source_column}" if m.source_column else ""
            def_str = f" (default: {m.default_value!r})" if m.default_value is not None else ""
            orphan_str = f" [orphan_policy: {m.orphan_policy}]" if m.orphan_policy else ""
            lines.append(f"- {m.new_column}{src_str} [transform: {trans_name}{def_str}]{orphan_str}: {doc}")

    return "\n".join(lines)


def get_sqlite_gotchas_block() -> str:
    """Generate concise SQLite gotchas block."""
    permitted = ", ".join(sorted(ALLOWED_PRAGMAS))
    return f"""SQLite Gotchas & Execution Rules:
- In 12-step table rebuilds, map orphan keys according to orphan_policy before inserting; never disable enforcement to hide violations. Always execute 'PRAGMA foreign_key_check;' before 'COMMIT;'.
- In 'CREATE TABLE target_new (...)', preserve the EXACT column nullability and defaults from the original schema for all untouched columns (e.g. if salary was nullable in the original schema, it MUST remain nullable; do NOT add NOT NULL to columns that were not originally NOT NULL).
- In 'INSERT INTO target_new (...) SELECT ... FROM target', explicitly list ALL columns in both INSERT and SELECT clauses (including untouched columns like id, salary, created_at). Never omit untouched columns or SQLite will fail NOT NULL constraints.
- If creating a FOREIGN KEY reference to a parent table (e.g. departments), map orphan foreign keys according to orphan_policy before inserting (e.g. map orphan codes to NULL or insert parent lookup record first); never disable enforcement to hide violations.
- Preserve the exact constraints (PRIMARY KEY, NOT NULL, DEFAULT) for all untouched columns from the original table schema.
- TRIM(x) only removes standard spaces; to strip tabs, newlines, and carriage returns, use: TRIM(x, ' ' || char(9) || char(10) || char(13))
- INSTR(string, substring) returns 0 when substring is not found (1-indexed).
- Concatenating with NULL produces NULL (e.g. 'a' || NULL is NULL); use COALESCE(col, '').
- SQLite has no built-in REGEXP operator.
- AUTOINCREMENT requires proper sqlite_sequence handling during table rebuilds.
- Recreate all indexes, triggers, and views after a table rebuild.
- Only the following permitted PRAGMAs may be used on the in-memory clone: {permitted}.
"""


def build_dynamic_task_checklist(result: InvariantResult | None) -> list[str]:
    """Dynamically construct task checklist from failed check types rather than scenario-specific advice."""
    checklist: list[str] = []
    if not result or not result.failed_checks:
        checklist.append("Ensure schema modifications match the approved specification.")
        checklist.append("Provide both complete UP migration and DOWN rollback migration.")
        return checklist

    failed_text = " ".join(result.failed_checks).upper()

    # 1. Row loss
    if "ROW_LOSS" in failed_text or "ROW_COUNT" in failed_text or "LOST_ROW" in failed_text or "ROW LOSS" in failed_text:
        checklist.append(
            "Fix row loss: ensure zero silent row drops by using LEFT JOIN and COALESCE so unmatched records or orphan foreign keys default safely without losing records."
        )

    # 2. Indexes
    if "INDEX" in failed_text or "LOST_INDEX" in failed_text:
        checklist.append("Recreate all pre-existing and declared indexes on the target table after rebuilding.")

    # 3. Triggers and Views
    if "TRIGGER" in failed_text or "VIEW" in failed_text:
        checklist.append("Recreate all pre-existing views and triggers that depend on the rebuilt table.")

    # 4. Data Mismatch
    if "DATA_MISMATCH" in failed_text or "MISMATCH" in failed_text or "ORACLE" in failed_text:
        checklist.append(
            "Fix data mismatch: compare actual output against the Transform Contract. Use TRIM(x, ' ' || char(9) || char(10) || char(13)) to handle whitespace and verify splitting boundaries."
        )

    # 5. Sandbox Violations
    if "SANDBOX VIOLATION" in failed_text or "NOT PERMITTED" in failed_text or "AUTHORIZ" in failed_text:
        permitted = ", ".join(sorted(ALLOWED_PRAGMAS))
        checklist.append(
            f"Fix sandbox violation: avoid ATTACH, DETACH, VACUUM, and load_extension. Only the following permitted PRAGMAs may be used: {permitted}."
        )

    # 6. Foreign Key Failures
    if "FOREIGN KEY" in failed_text or "FOREIGN_KEY" in failed_text:
        checklist.append(
            "Fix foreign key failure: map orphan keys according to orphan_policy before inserting; never disable enforcement to hide violations. Ensure orphan records (keys not in parent table) are safely mapped or defaulted before 'PRAGMA foreign_key_check;'."
        )

    # 7. Untouched Columns Mutation
    if "UNTOUCHED_COLUMN_MUTATED" in failed_text or "UNTOUCHED" in failed_text:
        checklist.append(
            "Fix untouched column mutation: do not alter, transform, or compute values for columns declared unchanged. Copy their original values directly without modification."
        )

    # 8. Schema Constraint Violations
    if "SCHEMA_CONSTRAINT_VIOLATION" in failed_text:
        checklist.append(
            "Fix schema constraint violation: check the specific column mentioned in the error. Preserve the exact column type, NULL/NOT NULL constraint, and PRIMARY KEY definitions from the original table for all preserved columns. If a column was nullable originally, define it as nullable."
        )

    # 9. NOT NULL or General Constraint Violations
    if "NOT NULL" in failed_text or "CONSTRAINT" in failed_text or "CHECK" in failed_text:
        import re
        nn_match = re.search(r"NOT NULL constraint failed:\s*([\w\.]+)", " ".join(result.failed_checks))
        col_name = nn_match.group(1) if nn_match else None
        if col_name:
            checklist.append(
                f"Fix NOT NULL constraint failure on '{col_name}': explicitly include '{col_name}' in BOTH the 'INSERT INTO target_new (..., {col_name}, ...)' column list and the 'SELECT ..., {col_name}, ... FROM target' data copy query. Check the original schema: if '{col_name}' was nullable originally, ensure it is nullable in target_new."
            )
        else:
            checklist.append(
                "Fix constraint violations: ensure all columns defined with NOT NULL constraints are explicitly included in the INSERT INTO column list and populated from the original table with non-null values or safe DEFAULT fallbacks."
            )

    # 10. General baseline requirement
    if not checklist:
        checklist.append("Fix the execution error or constraint violation reported in the failure telemetry.")

    checklist.append("Provide both complete UP migration and DOWN rollback migration inside ```sql blocks.")
    return checklist


def build_migration_prompt(
    schema: DatabaseSchema,
    spec: MigrationSpec,
    is_naive_baseline: bool = False,
    max_tool_calls: int = 6,
    profiles: dict[str, Any] | None = None,
) -> str:
    """Build prompt for generating up.sql and down.sql."""
    schema_text = format_schema_summary(schema)
    spec_json = spec.model_dump_json(indent=2)
    contract_text = generate_transform_contract(spec)
    gotchas_text = get_sqlite_gotchas_block()

    profile_section = ""
    if profiles:
        p_lines = []
        for tbl_name, col_profiles in profiles.items():
            for c_name, cp in col_profiles.items():
                nc = getattr(cp, "null_count", 0) if not isinstance(cp, dict) else cp.get("null_count", 0)
                tot = getattr(cp, "total_rows", 0) if not isinstance(cp, dict) else cp.get("total_rows", 0)
                ms = getattr(cp, "masked_sample", []) if not isinstance(cp, dict) else cp.get("masked_sample", [])
                sample_str = f" | Samples: {ms}" if ms else ""
                p_lines.append(f"- {tbl_name}.{c_name}: {nc}/{tot} NULL rows{sample_str}")
        if p_lines:
            profile_section = "\nData Profile & Masked Sample Facts:\n" + "\n".join(p_lines) + "\n"

    if is_naive_baseline:
        instruction = (
            "Write a standard first-pass naive migration script using INNER JOIN (as a baseline test).\n"
            "Output the migration script inside a ```sql ``` block."
        )
    else:
        instruction = (
            "Write a complete, safe, production-grade SQLite migration script.\n"
            "Requirements:\n"
            "1. Rebuild tables following SQLite standard 12-step table rebuild (create new table, copy data, drop old, rename, recreate indexes/triggers/views).\n"
            "2. Preserve all data records without silent drops.\n"
            "3. Output BOTH ```sql -- UP MIGRATION ``` and ```sql -- DOWN MIGRATION (ROLLBACK) ```.\n"
            f"4. The schema is already above; do not call get_schema. You have at most {max_tool_calls} tool calls; finish with submit.\n"
            "5. In the new table definition, preserve the EXACT column nullability and defaults from the original schema for all untouched columns (e.g. if salary was nullable originally, keep it nullable; do NOT add NOT NULL).\n"
            "6. In the INSERT INTO target_new (...) SELECT ... FROM target statement, explicitly list every column name in both the INSERT and SELECT clauses.\n"
            "7. When complete, call the submit tool or provide the migration scripts inside ```sql blocks."
        )

    contract_section = f"\n{contract_text}\n" if contract_text else ""

    return f"""You are a database migration engineer writing SQLite migration SQL.

Current Database Schema:
{schema_text}

Approved Migration Plan (MigrationSpec):
{spec_json}
{contract_section}{profile_section}
{gotchas_text}
{instruction}
"""


def format_previous_attempts_block(history: Any) -> str:
    """Format compact markdown table of previous attempts and cumulative requirements."""
    if not history or not getattr(history, "records", None):
        return ""

    lines = ["\n### Previous attempts (memory):"]
    lines.append("| Attempt | Model | Approach Summary | Failed Checks |")
    lines.append("|---|---|---|---|")
    for r in history.records:
        fc_str = "; ".join(r.failed_check_types[:2]) if r.failed_check_types else "None"
        appr_short = r.approach_summary[:60].replace("\n", " ") + ("..." if len(r.approach_summary) > 60 else "")
        lines.append(f"| {r.attempt_num} | {r.model} | {appr_short} | {fc_str} |")

    reqs = history.get_discovered_requirements() if hasattr(history, "get_discovered_requirements") else []
    if reqs:
        lines.append("\n### Requirements discovered so far (all must still hold):")
        for i, req in enumerate(reqs, 1):
            lines.append(f"{i}. {req}")

    return "\n".join(lines) + "\n"


def build_repair_prompt(
    schema: DatabaseSchema,
    spec: MigrationSpec,
    failed_sql: str,
    result: InvariantResult | None = None,
    trap_results: list[dict] | None = None,
    max_tool_calls: int = 6,
    history: Any = None,
    split_rule: str = "",
    down_sql: str = "",
) -> str:
    """Build repair prompt with structured failure telemetry, multi-attempt memory, and dynamic checklist."""
    schema_text = format_schema_summary(schema)
    contract_text = generate_transform_contract(spec)
    gotchas_text = get_sqlite_gotchas_block()

    tool_instruction = f"The schema is already above; do not call get_schema. You have at most {max_tool_calls} tool calls; finish with submit."

    history_block = format_previous_attempts_block(history) if history else ""

    reintroduced_warnings = ""
    if history and result and hasattr(history, "check_reintroduced_failures"):
        reintroduced = history.check_reintroduced_failures(result.failed_checks)
        if reintroduced:
            reintroduced_warnings = "\n" + "\n".join([f"⚠️ {w}" for w in reintroduced]) + "\n"

    parsed_status = f"Parsed Script Status: UP parsed (rule: {split_rule or 'standard'}), DOWN parsed: {bool(down_sql)}\n"

    if result is None:
        truncated_reply = (failed_sql or "")[:500]
        return f"""You are a database migration engineer writing SQLite migration SQL.

Target Schema:
{schema_text}

{history_block}
your previous reply contained no executable SQL (reply was: {truncated_reply}); reply with the migration.
{tool_instruction}

Output the complete, corrected SQLite migration inside a ```sql ``` block with both UP and DOWN migrations, or call submit."""

    failed_checks_text = "\n".join([f"- {fc}" for fc in result.failed_checks])
    traps_text = ""
    if trap_results:
        failing_traps = [t for t in trap_results if not t.get("survived")]
        if failing_traps:
            traps_text = "\nFailed Adversarial Traps:\n" + "\n".join(
                [f"- Trap {t.get('trap_type')}: {t.get('description')} (pk={t.get('pk')})" for t in failing_traps]
            )

    checklist_items = build_dynamic_task_checklist(result)
    checklist_text = "\n".join([f"{i + 1}. {item}" for i, item in enumerate(checklist_items)])
    contract_section = f"\n{contract_text}\n" if contract_text else ""

    return f"""The previous SQLite schema migration failed dry-run verification on the sandbox clone.
{history_block}
{reintroduced_warnings}
{parsed_status}
Failed SQL Script:
```sql
{failed_sql}
```

Sandbox Failure Telemetry:
Row Counts: Before={result.before_count}, After={result.after_count}
Failed Checks:
{failed_checks_text}
{traps_text}
{contract_section}
{gotchas_text}
Task:
Analyze the exact failure points above and write a complete, corrected, lossless SQLite migration.
{checklist_text}

{tool_instruction}
When complete, call the submit tool or return the corrected scripts in ```sql blocks with both UP and DOWN migrations."""
