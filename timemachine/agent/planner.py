"""Migration planning, prompt construction, JSON normalization, and spec generation."""

import json
import re
import time
from typing import Any, get_args

from openai import OpenAI
from pydantic import BaseModel

from timemachine.adapters.schema import DatabaseSchema
from timemachine.agent.llm import call_llm
from timemachine.agent.prompts import format_schema_summary
from timemachine.config import AgentError
from timemachine.logging import get_logger
from timemachine.verify.spec import MigrationSpec, TransformType

logger = get_logger("agent.planner")


class SpecResult(BaseModel):
    """Result of MigrationSpec generation."""
    spec: MigrationSpec | None = None
    source: str = "model"  # "model", "fallback", "error"
    raw_response: str = ""
    error: str | None = None
    model: str = ""
    latency_ms: float = 0.0


def build_spec_prompt(schema: DatabaseSchema, goal: str, profiles: dict | None = None) -> str:
    """Build prompt describing the database schema, data profiles, goal, and MigrationSpec JSONSchema."""
    schema_text = format_schema_summary(schema)
    profile_summary = ""
    if profiles:
        prof_dict = {}
        for k, v in profiles.items():
            if isinstance(v, dict):
                prof_dict[k] = {col: (cp.model_dump() if hasattr(cp, "model_dump") else cp) for col, cp in v.items()}
            else:
                prof_dict[k] = v.model_dump() if hasattr(v, "model_dump") else v
        profile_summary = f"\nData Profile Summary (Masked):\n{json.dumps(prof_dict, indent=2, default=str)}"

    json_schema_str = json.dumps(MigrationSpec.model_json_schema(), indent=2)

    return f"""You are an expert database migration architect.
Your task is to analyze the database schema and migration goal, then output a structured MigrationSpec JSON matching the schema below.

Database Schema:
{schema_text}
{profile_summary}

Migration Goal:
{goal}

Transform Contract & Whitespace Agreement:
The reference oracle normalizes string fields using Python str.strip() (stripping all ASCII and Unicode whitespace, including tabs and newlines).

Target JSON Schema:
{json_schema_str}

Rules for MigrationSpec formulation:
1. Column Mappings:
   - Only include new, renamed, or transformed columns in `column_mappings`.
   - Untouched / preserved columns (e.g. unchanged primary keys, dates, numbers) must be listed under `columns_declared_unchanged`.
   - Valid transform types: 'passthrough', 'split_name_first', 'split_name_last', 'strip_honorific', 'cast_int', 'cast_cents', 'default_if_null', 'map_via_lookup', 'normalize_date', 'salary_band', 'trim', 'lower', 'upper'.
2. Null Policy:
   - CRITICAL RULE FOR TABLE REBUILDS (CREATE TABLE ..._new): Preserve the exact nullability of every existing column from the original CREATE TABLE DDL. NEVER add NOT NULL to an existing nullable column (such as salary, created_at, etc.) unless the migration goal explicitly asks to make that specific column NOT NULL. Whenever any target column IS declared NOT NULL, you MUST wrap its corresponding SELECT expression in COALESCE(column_name, <sensible_default_matching_type>) for ALL such columns at once.
   - Set null_policy to 'not_null' ONLY for columns that are NOT NULL in the baseline schema or explicitly required to be non-nullable by the migration goal (such as split first_name and last_name).
   - Columns that are nullable in the source schema (e.g. department, salary, created_at) MUST have null_policy set to 'nullable'.
3. Constraints:
   - Only include foreign key constraints in `expected_constraints` if the migration goal explicitly requires adding foreign key constraints or if the baseline schema already contained them.

Output ONLY the MigrationSpec JSON object inside a ```json ``` block."""


def extract_json_from_text(raw_text: str) -> dict[str, Any]:
    """Extract and parse JSON object from LLM response text."""
    # 1. Strip <think>...</think> and unclosed <think>...
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()

    # 2. Check for fenced JSON blocks
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        return json.loads(cleaned)
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
        return json.loads(cleaned)

    # 3. If there is no fence, parse the first balanced {...} with json.JSONDecoder().raw_decode
    start_idx = cleaned.find("{")
    if start_idx != -1:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(cleaned[start_idx:])
        if isinstance(obj, dict):
            return obj
        raise ValueError(f"Decoded JSON is not an object: {type(obj)}")

    return json.loads(cleaned)


def normalize_plan_dict(plan_dict: dict[str, Any], target_table: str = "employees") -> dict[str, Any]:
    """Normalize raw dictionary to conform to MigrationSpec requirements.

    - Convert list-shaped column_mappings to dict keyed by target table.
    - Coerce risky_operations items to str.
    - Coerce unknown transform values to 'custom'.
    - Log and drop extra unknown keys.
    """
    valid_fields = set(MigrationSpec.model_fields.keys())
    dropped_keys = [k for k in plan_dict.keys() if k not in valid_fields]
    if dropped_keys:
        logger.info(f"Dropped extra keys not in MigrationSpec: {dropped_keys}")
        for k in dropped_keys:
            del plan_dict[k]

    # Determine default table key
    affected = plan_dict.get("affected_tables")
    default_key = affected[0] if (isinstance(affected, list) and len(affected) > 0 and isinstance(affected[0], str)) else target_table

    # 1. Normalize column_mappings: if list-shaped -> dict keyed by target table
    col_mappings = plan_dict.get("column_mappings")
    if isinstance(col_mappings, list):
        plan_dict["column_mappings"] = {default_key: col_mappings}

    # 2. Normalize transform in column_mappings to documented set (unknown -> 'custom')
    valid_transforms = set(get_args(TransformType))
    if isinstance(plan_dict.get("column_mappings"), dict):
        for tbl, mappings in plan_dict["column_mappings"].items():
            if isinstance(mappings, list):
                for m in mappings:
                    if isinstance(m, dict):
                        trans = m.get("transform")
                        if trans not in valid_transforms:
                            m["transform"] = "custom"
                        np = str(m.get("null_policy", "")).lower()
                        if np not in ("not_null", "nullable"):
                            m["null_policy"] = "nullable"

    # 3. Coerce risky_operations items to str
    risky = plan_dict.get("risky_operations")
    if isinstance(risky, list):
        plan_dict["risky_operations"] = [
            (item.get("description") or item.get("action") or str(item)) if isinstance(item, dict) else str(item)
            for item in risky
        ]

    return plan_dict


def auto_fix_spec_null_policies(spec: MigrationSpec, schema: DatabaseSchema) -> None:
    """Auto-fix null_policy on mappings where appropriate.

    - A passthrough/unchanged mapping on a source column that is nullable
      must get null_policy="nullable" (auto-fix and log), unless the transform
      supplies a default (default_if_null, split_name_last default, cast_int default).
    - An INTEGER PRIMARY KEY column never requires NOT NULL (SQLite ROWID alias).
    """
    for tbl_name, mappings in spec.column_mappings.items():
        tbl_info = schema.get_table(tbl_name)
        if not tbl_info:
            continue
        col_by_name = {c.name.lower(): c for c in tbl_info.columns}
        unchanged_cols = {c.lower() for c in spec.columns_declared_unchanged.get(tbl_name, [])}

        for m in mappings:
            src_col_name = (m.source_column or m.new_column).lower()
            src_col = col_by_name.get(src_col_name)

            # Never require NOT NULL for an INTEGER PRIMARY KEY
            if (src_col and src_col.pk > 0 and "INT" in src_col.type.upper()) or (m.new_column.lower() == "id" and src_col and src_col.pk > 0):
                if m.null_policy == "not_null":
                    logger.info(
                        f"Auto-fixing null_policy for INTEGER PRIMARY KEY column '{tbl_name}.{m.new_column}' to 'nullable'."
                    )
                    m.null_policy = "nullable"
                continue

            # Check if passthrough or declared unchanged
            is_unchanged = m.new_column.lower() in unchanged_cols or src_col_name in unchanged_cols
            is_passthrough = (
                m.transform in ("passthrough", "custom", "identity", None)
                or is_unchanged
            )

            # Check if transform supplies a default
            has_default = (
                m.default_value is not None
                or m.transform in ("default_if_null", "split_name_last", "cast_int")
            )

            if is_passthrough and src_col is not None:
                # If source column is nullable or if declared unchanged
                if (not src_col.notnull or is_unchanged) and not has_default and m.null_policy == "not_null":
                    logger.info(
                        f"Auto-fixing null_policy for nullable passthrough column '{tbl_name}.{m.new_column}' to 'nullable'."
                    )
                    m.null_policy = "nullable"


def validate_spec_against_schema(
    spec: MigrationSpec,
    schema: DatabaseSchema,
    target_table: str,
    profiles: dict[str, Any] | None = None,
) -> list[str]:
    """Validate MigrationSpec against the actual database schema.

    Checks:
    1. target_table is in affected_tables
    2. each affected table exists (or is explicitly declared as created by the migration)
    3. every source_column exists in the source table (or is None)
    4. lookup_table, lookup_key and lookup_val exist
    5. columns_declared_unchanged and expected_constraints reference real columns
    6. mapping is not_null, source profile contains NULLs, and no default exists
    """
    errors: list[str] = []

    # 1. target_table is in affected_tables
    if target_table not in spec.affected_tables:
        errors.append(
            f"Target table '{target_table}' is not in affected_tables: {spec.affected_tables}"
        )

    # 2. each affected table exists (or is explicitly declared as created by the migration)
    created_tables: set[str] = set()
    for op in spec.risky_operations:
        m = re.search(r"\b(?:create|creates|created|new)\s+(?:table\s+)?([a-zA-Z0-9_]+)\b", str(op), re.IGNORECASE)
        if m:
            created_tables.add(m.group(1).lower())
    if spec.goal:
        m_goal = re.findall(r"\b(?:create|creates|created|new)\s+(?:table\s+)?([a-zA-Z0-9_]+)\b", spec.goal, re.IGNORECASE)
        for g_tbl in m_goal:
            created_tables.add(g_tbl.lower())

    for tbl in spec.affected_tables:
        if tbl not in schema.tables and tbl.lower() not in created_tables:
            errors.append(
                f"Affected table '{tbl}' does not exist in schema and is not declared as created by migration"
            )

    # 3. every source_column exists in the source table (or is None)
    for tbl_name, mappings in spec.column_mappings.items():
        if tbl_name in schema.tables:
            tbl_info = schema.get_table(tbl_name)
            col_names = {c.name for c in tbl_info.columns} if tbl_info else set()
            for m in mappings:
                if m.source_column is not None and m.source_column not in col_names:
                    errors.append(
                        f"ColumnMapping for '{tbl_name}.{m.new_column}' references nonexistent source_column '{m.source_column}'"
                    )

    # 4. lookup_table, lookup_key and lookup_val exist
    for tbl_name, mappings in spec.column_mappings.items():
        for m in mappings:
            if m.lookup_table or m.lookup_key or m.lookup_val or m.transform == "map_via_lookup":
                if m.lookup_table:
                    if m.lookup_table not in schema.tables and m.lookup_table.lower() not in created_tables:
                        errors.append(
                            f"Lookup table '{m.lookup_table}' in mapping for '{m.new_column}' does not exist in schema"
                        )
                    else:
                        lt_info = schema.get_table(m.lookup_table)
                        lt_cols = {c.name for c in lt_info.columns} if lt_info else set()
                        if m.lookup_key and m.lookup_key not in lt_cols:
                            errors.append(
                                f"Lookup key '{m.lookup_key}' does not exist in lookup table '{m.lookup_table}'"
                            )
                        if m.lookup_val and m.lookup_val not in lt_cols:
                            errors.append(
                                f"Lookup val '{m.lookup_val}' does not exist in lookup table '{m.lookup_table}'"
                            )

    # 5. columns_declared_unchanged and expected_constraints reference real columns
    SQL_KEYWORDS = {
        "PRIMARY", "KEY", "NOT", "NULL", "UNIQUE", "CHECK", "FOREIGN",
        "REFERENCES", "INTEGER", "TEXT", "REAL", "BLOB", "DEFAULT",
        "AUTOINCREMENT", "AND", "OR", "ASC", "DESC",
    }
    for tbl_name, cols in spec.columns_declared_unchanged.items():
        if tbl_name in schema.tables:
            tbl_info = schema.get_table(tbl_name)
            col_names = {c.name for c in tbl_info.columns} if tbl_info else set()
            for col in cols:
                if col not in col_names:
                    errors.append(
                        f"columns_declared_unchanged specifies '{col}' which does not exist in table '{tbl_name}'"
                    )

    for tbl_name, constraints in spec.expected_constraints.items():
        if tbl_name in schema.tables:
            tbl_info = schema.get_table(tbl_name)
            col_names = {c.name for c in tbl_info.columns} if tbl_info else set()
            new_cols = {m.new_column for m in spec.get_table_mappings(tbl_name)}
            valid_cols = col_names | new_cols
            for item in constraints:
                words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", item)
                col_words = [w for w in words if w.upper() not in SQL_KEYWORDS]
                if col_words:
                    if not any(w in valid_cols for w in col_words):
                        errors.append(
                            f"expected_constraints for table '{tbl_name}' references nonexistent column in '{item}'"
                        )
                elif item not in valid_cols:
                    errors.append(
                        f"expected_constraints for table '{tbl_name}' references nonexistent column '{item}'"
                    )

    # 6. Check NOT NULL with source profile NULLs
    if profiles:
        for tbl_name, mappings in spec.column_mappings.items():
            tbl_profiles = profiles.get(tbl_name, {})
            for m in mappings:
                if m.null_policy == "not_null":
                    src_col_name = m.source_column or m.new_column
                    prof = tbl_profiles.get(src_col_name) if isinstance(tbl_profiles, dict) else None
                    null_count = 0
                    if prof is not None:
                        if isinstance(prof, dict):
                            null_count = prof.get("null_count", 0)
                        else:
                            null_count = getattr(prof, "null_count", 0)
                    has_default = (
                        m.default_value is not None
                        or m.transform in ("default_if_null", "split_name_last", "cast_int")
                    )
                    if null_count > 0 and not has_default:
                        errors.append(
                            f"Column '{tbl_name}.{m.new_column}' is specified as NOT NULL, but source data in '{src_col_name}' contains {null_count} NULLs and no default is provided."
                        )

    return errors


def generate_spec(
    client: OpenAI,
    model: str,
    schema: DatabaseSchema,
    goal: str,
    profiles: dict | None = None,
    target_table: str = "employees",
) -> SpecResult:
    """Generate, validate, and normalize a structured MigrationSpec using the model."""
    start_time = time.time()
    prompt = build_spec_prompt(schema, goal, profiles)
    messages = [{"role": "user", "content": prompt}]

    raw_response = ""
    error_str = None
    spec = None

    try:
        raw_msg, _ = call_llm(client, model, messages=messages, temperature=0.1)
        raw_response = raw_msg.content or ""
        plan_dict = extract_json_from_text(raw_response)
        normalized = normalize_plan_dict(plan_dict, target_table=target_table)
        candidate = MigrationSpec.model_validate(normalized)
        auto_fix_spec_null_policies(candidate, schema)
        val_errs = validate_spec_against_schema(candidate, schema, target_table, profiles=profiles)
        if val_errs:
            raise ValueError(f"Schema validation failed: {'; '.join(val_errs)}")
        spec = candidate
    except AgentError as ae:
        logger.error(f"Planning failed due to AgentError: {ae}")
        return SpecResult(
            spec=None,
            source="error",
            raw_response=raw_response,
            error=str(ae),
            model=model,
            latency_ms=(time.time() - start_time) * 1000,
        )
    except Exception as first_exc:
        if not raw_response.strip():
            logger.error(f"Model returned empty response for plan: {first_exc}")
            return SpecResult(
                spec=None,
                source="error",
                raw_response="",
                error=f"Model returned empty response: {first_exc}",
                model=model,
                latency_ms=(time.time() - start_time) * 1000,
            )
        logger.warning(f"Initial MigrationSpec parsing failed: {first_exc}. Retrying once with error feedback...")
        # Retry once with pydantic error appended
        retry_prompt = (
            f"Your previous output failed validation with the following error:\n{first_exc}\n"
            f"Please output a corrected, valid MigrationSpec JSON matching the schema."
        )
        retry_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw_response},
            {"role": "user", "content": retry_prompt},
        ]
        try:
            retry_msg, _ = call_llm(client, model, messages=retry_messages, temperature=0.1)
            raw_response = retry_msg.content or ""
            plan_dict = extract_json_from_text(raw_response)
            normalized = normalize_plan_dict(plan_dict, target_table=target_table)
            candidate = MigrationSpec.model_validate(normalized)
            auto_fix_spec_null_policies(candidate, schema)
            val_errs = validate_spec_against_schema(candidate, schema, target_table, profiles=profiles)
            if val_errs:
                raise ValueError(f"Schema validation failed after retry: {'; '.join(val_errs)}")
            spec = candidate
        except AgentError as ae:
            logger.error(f"Plan retry failed due to AgentError: {ae}")
            return SpecResult(
                spec=None,
                source="error",
                raw_response=raw_response,
                error=str(ae),
                model=model,
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as retry_exc:
            error_str = f"Plan validation failed after retry: {retry_exc}"
            logger.error(error_str)

    latency_ms = (time.time() - start_time) * 1000

    if spec is not None:
        return SpecResult(
            spec=spec,
            source="model",
            raw_response=raw_response,
            error=None,
            model=model,
            latency_ms=latency_ms,
        )
    else:
        return SpecResult(
            spec=None,
            source="error",
            raw_response=raw_response,
            error=error_str or "Failed to generate spec",
            model=model,
            latency_ms=latency_ms,
        )
