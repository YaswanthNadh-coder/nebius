"""Migration planning, prompt construction, JSON normalization, and spec generation."""

import json
import time
from typing import Any, get_args

from openai import OpenAI
from pydantic import BaseModel

from timemachine.adapters.schema import DatabaseSchema
from timemachine.agent.llm import call_llm
from timemachine.agent.prompts import format_schema_summary
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

Target JSON Schema:
{json_schema_str}

Output ONLY the MigrationSpec JSON object inside a ```json ``` block."""


def extract_json_from_text(raw_text: str) -> dict[str, Any]:
    """Extract and parse JSON object from LLM response text."""
    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
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

    # 3. Coerce risky_operations items to str
    risky = plan_dict.get("risky_operations")
    if isinstance(risky, list):
        plan_dict["risky_operations"] = [
            (item.get("description") or item.get("action") or str(item)) if isinstance(item, dict) else str(item)
            for item in risky
        ]

    return plan_dict


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
        spec = MigrationSpec.model_validate(normalized)
    except Exception as first_exc:
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
            spec = MigrationSpec.model_validate(normalized)
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
