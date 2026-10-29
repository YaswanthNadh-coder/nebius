"""Autonomous agent loop with tool-calling, tiered model escalation, and structured telemetry."""

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from timemachine.adapters.base import DbAdapter
from timemachine.agent.llm import call_llm, clean_sql, get_client, resolve_nebius_models
from timemachine.agent.planner import (
    auto_fix_spec_null_policies,
    generate_spec,
    validate_spec_against_schema,
)
from timemachine.agent.prompts import (
    build_migration_prompt,
    build_repair_prompt,
    format_schema_summary,
    generate_transform_contract,
    get_sqlite_gotchas_block,
)
from timemachine.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from timemachine.config import AgentError, get_settings
from timemachine.logging import get_logger
from timemachine.traps.generator import audit_traps
from timemachine.traps.profiler import (
    get_cached_profile,
    profile_database,
    set_cached_profile,
)
from timemachine.verify.invariants import (
    InvariantResult,
    audit_crash_diagnostics,
    evaluate_migration_invariants,
)
from timemachine.verify.rebuild import (
    estimate_production_lock_time,
    sanitize_down_script,
    verify_rollback_roundtrip,
)
from timemachine.verify.spec import MigrationSpec

logger = get_logger("agent.loop")


class AgentStepLog(BaseModel):
    """Log record for an individual agent execution step."""
    step: int
    model: str
    action: str
    details: dict[str, Any] = Field(default_factory=dict)
    tokens_used: int = 0
    latency_ms: float = 0.0
    status: str = "ok"


class AttemptRecord(BaseModel):
    """Record of an individual migration attempt and its verification outcome."""
    attempt: int
    model: str
    passed_count: int = 0
    failed_count: int = 0
    first_failed_check: str = ""
    sandbox_error: str | None = None
    latency_ms: float = 0.0
    tokens: int = 0
    raw_reply: str = ""
    up_sql: str = ""
    down_sql: str = ""
    tool_rounds_used: int = 0
    no_sql_replies: int = 0
    split_rule: str = ""


class AttemptHistoryRecord(BaseModel):
    """Compact history record for multi-attempt memory."""
    attempt_num: int
    model: str
    sql_hash: str
    approach_summary: str
    failed_check_types: list[str] = Field(default_factory=list)
    failed_check_messages: list[str] = Field(default_factory=list)
    passed_check_types: list[str] = Field(default_factory=list)
    up_sql: str = ""
    down_sql: str = ""
    split_rule: str = ""


class AttemptHistory:
    """Tracks history across repair attempts to prevent regressions and accumulate requirements."""

    def __init__(self):
        self.records: list[AttemptHistoryRecord] = []
        self._all_seen_failures: dict[str, str] = {}  # check_type -> requirement text

    @staticmethod
    def normalize_check_type(check_msg: str) -> str:
        """Extract a canonical check category from a failure message."""
        upper = check_msg.upper()
        if "NOT NULL" in upper:
            m = re.search(r"NOT NULL constraint failed:\s*([\w\.]+)", check_msg, re.IGNORECASE)
            col = m.group(1).split(".")[-1] if m else "general"
            return f"NOT_NULL:{col}"
        if "UNTOUCHED_COLUMN_MUTATED" in upper or "UNTOUCHED" in upper:
            m = re.search(r"Column '([\w]+)'", check_msg)
            col = m.group(1) if m else "general"
            return f"UNTOUCHED:{col}"
        if "SCHEMA_CONSTRAINT_VIOLATION" in upper:
            m = re.search(r"Column '([\w]+)'", check_msg)
            col = m.group(1) if m else "general"
            return f"SCHEMA_CONSTRAINT:{col}"
        if "FOREIGN KEY" in upper or "FOREIGN_KEY" in upper:
            return "FOREIGN_KEY"
        if "DATA_MISMATCH" in upper or "MISMATCH" in upper:
            m = re.search(r"([\w]+):\s*\d+\s*wrong", check_msg)
            col = m.group(1) if m else "transform"
            return f"DATA_MISMATCH:{col}"
        if "LOST_INDEX" in upper:
            m = re.search(r"Index '([\w]+)'", check_msg)
            idx = m.group(1) if m else "index"
            return f"LOST_INDEX:{idx}"
        if "LOST_TRIGGER" in upper:
            m = re.search(r"Trigger '([\w]+)'", check_msg)
            trg = m.group(1) if m else "trigger"
            return f"LOST_TRIGGER:{trg}"
        if "LOST_VIEW" in upper:
            m = re.search(r"View '([\w]+)'", check_msg)
            vw = m.group(1) if m else "view"
            return f"LOST_VIEW:{vw}"
        if "ROW_LOSS" in upper or "SILENT_DATA_LOSS" in upper:
            return "ROW_LOSS"
        if "UP_UNDONE" in upper:
            return "UP_UNDONE"
        return upper.split(":")[0].strip() if ":" in upper else upper[:30].strip()

    def record_attempt(
        self,
        attempt_num: int,
        model: str,
        up_sql: str,
        inv_result: InvariantResult | None,
        down_sql: str = "",
        split_rule: str = "",
    ) -> None:
        """Record an attempt's telemetry and failures."""
        sql_hash = hashlib.sha256(normalize_sql_for_progress(up_sql).encode()).hexdigest()[:8] if up_sql else "none"

        # Construct 200-char approach summary from the UP SQL
        sql_clean = re.sub(r"\s+", " ", up_sql or "").strip()
        approach = sql_clean[:200] if sql_clean else "No executable SQL"

        failed_types: list[str] = []
        failed_msgs: list[str] = []
        passed_types: list[str] = []

        if inv_result:
            for fc in inv_result.failed_checks:
                ctype = self.normalize_check_type(fc)
                failed_types.append(ctype)
                failed_msgs.append(fc)
                # Store discovered requirement
                if ctype not in self._all_seen_failures:
                    self._all_seen_failures[ctype] = self._derive_requirement(ctype, fc)
            for pc in inv_result.passed_checks:
                passed_types.append(pc)

        rec = AttemptHistoryRecord(
            attempt_num=attempt_num,
            model=model,
            sql_hash=sql_hash,
            approach_summary=approach,
            failed_check_types=failed_types,
            failed_check_messages=failed_msgs,
            passed_check_types=passed_types,
            up_sql=up_sql,
            down_sql=down_sql,
            split_rule=split_rule,
        )
        self.records.append(rec)

    def _derive_requirement(self, ctype: str, msg: str) -> str:
        """Derive actionable rule requirement from failure type."""
        if ctype.startswith("NOT_NULL:"):
            col = ctype.split(":", 1)[1]
            return f"Column '{col}' must be explicitly included in INSERT/SELECT or declared nullable if nullable in baseline."
        if ctype.startswith("UNTOUCHED:"):
            col = ctype.split(":", 1)[1]
            return f"Column '{col}' declared unchanged must be copied directly without transformation or mutation."
        if ctype.startswith("SCHEMA_CONSTRAINT:"):
            col = ctype.split(":", 1)[1]
            return f"Column '{col}' schema definition must preserve original constraints and nullability."
        if ctype == "FOREIGN_KEY":
            return "Orphan foreign keys must be mapped or inserted into the parent table before PRAGMA foreign_key_check."
        if ctype.startswith("DATA_MISMATCH:"):
            col = ctype.split(":", 1)[1]
            return f"Transformation for '{col}' must match Transform Contract (strip tabs and carriage returns)."
        if ctype.startswith("LOST_INDEX:"):
            idx = ctype.split(":", 1)[1]
            return f"Index '{idx}' must be recreated after table rename."
        if ctype.startswith("LOST_TRIGGER:"):
            trg = ctype.split(":", 1)[1]
            return f"Trigger '{trg}' must be recreated after table rename."
        if ctype == "UP_UNDONE":
            return "Do not include rollback / undo statements inside up_sql; keep UP and DOWN strictly separated."
        return f"Resolve check: {msg[:100]}"

    def get_discovered_requirements(self) -> list[str]:
        """Return cumulative union of all requirements discovered across attempts."""
        return list(self._all_seen_failures.values())

    def check_reintroduced_failures(self, current_failed_checks: list[str]) -> list[str]:
        """Detect if any failure in the current attempt was previously fixed in an earlier attempt."""
        if len(self.records) < 2:
            return []

        reintroduced: list[str] = []
        prev_rec = self.records[-1]  # immediate previous attempt
        prev_failed_types = set(prev_rec.failed_check_types)

        # Look at attempts before the immediate previous one
        earlier_failed_types = set()
        for r in self.records[:-1]:
            earlier_failed_types.update(r.failed_check_types)

        for fc in current_failed_checks:
            ctype = self.normalize_check_type(fc)
            # If it failed earlier, was fixed in the immediate previous attempt, and has failed again:
            if ctype in earlier_failed_types and ctype not in prev_failed_types:
                reintroduced.append(
                    f"Warning: Failure on '{ctype}' was resolved in attempt {prev_rec.attempt_num} but reintroduced now! Maintain all previously established fixes."
                )

        return reintroduced


class AgentRunResult(BaseModel):
    """Final outcome of an autonomous migration dry-run execution."""
    status: str  # VERIFIED, UNVERIFIED, FAILED
    up_sql: str = ""
    down_sql: str = ""
    spec: MigrationSpec | None = None
    steps: list[AgentStepLog] = Field(default_factory=list)
    attempts: list[AttemptRecord] = Field(default_factory=list)
    invariant_result: InvariantResult | None = None
    rollback_verified: bool = False
    rollback_error: str | None = None
    lossy_columns: list[str] = Field(default_factory=list)
    rollback_classification: str = ""
    lock_estimate: dict[str, Any] = Field(default_factory=dict)
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    stop_reason: str = ""
    notes: str = ""
    fast_model: str = ""
    ultra_model: str = ""
    model_warnings: list[str] = Field(default_factory=list)


def has_ddl_or_dml(sql: str) -> bool:
    """Check if SQL contains any DDL or DML statements (not just queries or empty checks)."""
    if not sql:
        return False
    s = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)
    return bool(re.search(r"\b(CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|REPLACE|TRUNCATE)\b", s, re.IGNORECASE))


def normalize_sql_for_progress(sql: str) -> str:
    """Normalize SQL for progress detection: strip comments, collapse whitespace, lowercase."""
    s = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _clean_extracted_block(block: str) -> str:
    """Clean extracted SQL block by stripping think tags, stray fences, and extra whitespace."""
    if not block:
        return ""
    t = block.replace("\r\n", "\n")
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
    t = re.sub(r"<think>.*", "", t, flags=re.DOTALL)
    t = re.sub(r"^```.*$", "", t, flags=re.MULTILINE)
    return t.strip()


def _clean_sql_section(section: str) -> str:
    """Clean a SQL section from free-form text or stripped fences."""
    cleaned = _clean_extracted_block(section)
    res = clean_sql(cleaned)
    return res if res else cleaned


def strip_thinking(text: str | None) -> str:
    """Strip <think>...</think> reasoning blocks from text to keep conversation history concise."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def truncate_older_tool_results(
    messages: list[dict[str, Any]],
    keep_full_count: int = 2,
    max_truncated_chars: int = 200,
) -> list[dict[str, Any]]:
    """Truncate older tool results in message history, keeping the last `keep_full_count` full."""
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep_full_count:
        return messages

    indices_to_truncate = set(tool_indices[:-keep_full_count])
    truncated_messages = []
    for i, m in enumerate(messages):
        if i in indices_to_truncate:
            content = m.get("content", "")
            if len(content) > max_truncated_chars:
                m_copy = dict(m)
                m_copy["content"] = content[:max_truncated_chars] + "... [earlier tool output truncated to save context]"
                truncated_messages.append(m_copy)
            else:
                truncated_messages.append(m)
        else:
            truncated_messages.append(m)
    return truncated_messages


def parse_sql_scripts_ex(content: str) -> tuple[str, str, str]:
    """Parse UP and optional DOWN migration scripts from LLM content with rule tracking.

    Returns:
        (up_sql, down_sql, rule_matched)
    """
    text = content.replace("\r\n", "\n")
    down_keywords = r"(?:down|rollback|revert|undo|reverse|backward|migrate[ \t]+down)"
    banner_chars = r"[-=*~_# \t]"

    down_line_pattern = re.compile(
        r"(?im)^[ \t]*(?:"
        r"(?:```[a-zA-Z0-9_-]*[ \t]*)?"
        r"(?:--|/\*|#+)" + banner_chars + r"*\b" + down_keywords + r"\b" + banner_chars + r"*(?:\*/)?[^\r\n]*|"
        r"\*\*" + banner_chars + r"*\b" + down_keywords + r"\b" + banner_chars + r"*:?[ \t]*\*\*[^\r\n]*|"
        r"[-=*~_#]{2,}[ \t]*\b" + down_keywords + r"\b[ \t]*[-=*~_#]{2,}[^\r\n]*"
        r")$"
    )
    down_tag_pattern = re.compile(r"(?i)\b" + down_keywords + r"\b")

    # 1. Check for fenced blocks first
    fence_pattern = re.compile(r"(?:^|\n)```([^\n]*)\n(.*?)\n```", re.DOTALL)
    matches = list(fence_pattern.finditer(text))

    if matches:
        # Check single fence internal split
        if len(matches) == 1:
            body = matches[0].group(2)
            internal_match = down_line_pattern.search(body)
            if internal_match:
                up_part = body[:internal_match.start()]
                down_part = body[internal_match.end():]
                up = _clean_sql_section(up_part)
                down = _clean_sql_section(down_part)
                if has_ddl_or_dml(down):
                    return up, down, "single_fence_banner_split"

        down_start_idx = None
        matched_rule = ""
        # IMPORTANT: prose before the FIRST block must NEVER mark block 0 as DOWN.
        for i in range(1, len(matches)):
            tag = matches[i].group(1).strip()
            block_body = matches[i].group(2).strip()
            prev_end = matches[i - 1].end()
            prose = text[prev_end:matches[i].start()].strip()

            if down_tag_pattern.search(tag):
                down_start_idx = i
                matched_rule = f"fence_tag_{tag}"
                break
            first_line = block_body.split("\n", 1)[0] if block_body else ""
            if down_line_pattern.search(first_line):
                down_start_idx = i
                matched_rule = "code_first_line_comment"
                break
            if down_line_pattern.search(prose) or down_tag_pattern.search(prose):
                down_start_idx = i
                matched_rule = "prose_marker_before_fence"
                break

        if down_start_idx is not None and down_start_idx > 0:
            up_blocks = [_clean_extracted_block(matches[j].group(2)) for j in range(down_start_idx) if matches[j].group(2).strip()]
            down_blocks = [_clean_extracted_block(matches[j].group(2)) for j in range(down_start_idx, len(matches)) if matches[j].group(2).strip()]
            up = "\n\n".join(b for b in up_blocks if b)
            down = "\n\n".join(b for b in down_blocks if b)
            if not has_ddl_or_dml(down):
                down = ""
            return up, down, matched_rule

        # If 2+ blocks and no inter-fence marker, check if block 0 has an internal banner
        if len(matches) > 1:
            body0 = matches[0].group(2)
            m0 = down_line_pattern.search(body0)
            if m0:
                up_part = body0[:m0.start()]
                down_part = body0[m0.end():]
                up = _clean_sql_section(up_part)
                down = _clean_sql_section(down_part)
                if has_ddl_or_dml(down):
                    return up, down, "first_fence_banner_split"

        if len(matches) >= 2:
            all_blocks = [_clean_extracted_block(m.group(2)) for m in matches if m.group(2).strip()]
            up = "\n\n".join(b for b in all_blocks if b)
            return up, "", "multiple_blocks_no_marker"

    # 2. Split on a down/rollback line without code fences
    match = down_line_pattern.search(text)
    if match:
        up_part = text[:match.start()]
        down_part = text[match.end():]
        up = _clean_sql_section(up_part)
        down = _clean_sql_section(down_part)
        if has_ddl_or_dml(down):
            return up, down, "text_line_marker_split"

    if matches:
        all_blocks = [_clean_extracted_block(m.group(2)) for m in matches if m.group(2).strip()]
        up = "\n\n".join(b for b in all_blocks if b)
        return up, "", "fenced_blocks_up_only"

    return clean_sql(text), "", "fallback_clean_sql"


def parse_sql_scripts(content: str) -> tuple[str, str]:
    """Parse UP and optional DOWN migration scripts from LLM content without guessing by position."""
    up, down, _ = parse_sql_scripts_ex(content)
    return up, down


def get_model_timeout(model: str, settings: Any) -> float:
    """Return model-specific request timeout in seconds."""
    m_lower = model.lower()
    if "ultra" in m_lower:
        val = getattr(settings, "ultra_llm_timeout_s", None)
        if isinstance(val, (int, float)):
            return max(75.0, float(val))
        return 75.0
    elif "super" in m_lower:
        val = getattr(settings, "fast_llm_timeout_s", None)
        if isinstance(val, (int, float)):
            return max(45.0, float(val))
        return 45.0

    val = getattr(settings, "llm_timeout_s", None)
    if isinstance(val, (int, float)):
        return max(45.0, float(val))
    return 75.0


def run_time_machine_agent(
    adapter: DbAdapter,
    goal: str,
    spec: MigrationSpec | None = None,
    target_table: str = "employees",
    api_key_override: str | None = None,
    max_steps: int | None = None,
    is_naive_baseline: bool = False,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    wall_clock_timeout_s: float | None = None,
) -> AgentRunResult:
    """Run autonomous dry-run migration agent loop.

    1. Planning: Generates or uses approved MigrationSpec.
    2. Drafting: Super generates initial up_sql & down_sql or uses tool calling.
    3. Dry-Run Verification: Executes up_sql in disposable sandbox, runs deep invariants & oracles.
    4. Escalation: If verification fails, escalates to Ultra with structured telemetry.
    5. Rollback Verification: Proves up_sql -> down_sql round trip on disposable clone.
    """
    client = get_client(api_key_override)
    fast_model, ultra_model, warnings = resolve_nebius_models(client)

    start_time = time.time()
    steps: list[AgentStepLog] = []
    total_tokens = 0

    def notify(msg: str, data: dict[str, Any] | None = None):
        if progress_callback:
            progress_callback(msg, data or {})

    # Introspect source schema
    schema = adapter.introspect()
    tbl = schema.get_table(target_table)
    prod_rows = tbl.row_count if tbl else 1000

    fp = getattr(adapter, "source_fingerprint", "default") if hasattr(adapter, "source_fingerprint") else "default"
    traps_enabled = getattr(adapter, "inject_traps", True) if hasattr(adapter, "inject_traps") else True
    profiles: dict[str, Any] = get_cached_profile(fp, traps_enabled) or {}

    # Phase 1: Planning (derive spec if not provided, or validate supplied spec)
    if spec is not None:
        try:
            auto_fix_spec_null_policies(spec, schema)
        except Exception as af_err:
            logger.debug(f"Auto-fix null policies notice: {af_err}")
        val_errors = validate_spec_against_schema(spec, schema, target_table)
        if val_errors:
            logger.error(f"Supplied MigrationSpec failed schema validation: {val_errors}")
            return AgentRunResult(
                status="FAILED",
                up_sql="",
                down_sql="",
                spec=spec,
                steps=steps,
                total_tokens=total_tokens,
                total_latency_ms=(time.time() - start_time) * 1000,
                stop_reason="SPEC_INVALID",
                notes=f"Specification invalid for current schema: {'; '.join(val_errors)}",
                fast_model=fast_model,
                ultra_model=ultra_model,
                model_warnings=warnings,
            )
    else:
        notify(f"Generating MigrationSpec using {fast_model}...")
        if not profiles:
            clone = adapter.clone()
            try:
                profiles = profile_database(clone, schema)
                set_cached_profile(fp, traps_enabled, profiles)
            except Exception as exc:
                logger.debug(f"Could not compute database profile: {exc}")
                profiles = {}
            finally:
                clone.close()

        spec_result = generate_spec(
            client=client,
            model=fast_model,
            schema=schema,
            goal=goal,
            profiles=profiles,
            target_table=target_table,
        )

        if spec_result.spec is not None:
            spec = spec_result.spec
            steps.append(
                AgentStepLog(
                    step=len(steps) + 1,
                    model=fast_model,
                    action="generate_spec",
                    details={"spec": spec.model_dump(), "source": spec_result.source},
                    latency_ms=spec_result.latency_ms,
                    status="ok",
                )
            )
            notify("MigrationSpec formulated successfully.", {"spec": spec.model_dump()})
        else:
            logger.error(f"Failed to generate MigrationSpec: {spec_result.error}")
            steps.append(
                AgentStepLog(
                    step=len(steps) + 1,
                    model=fast_model,
                    action="generate_spec",
                    details={"error": spec_result.error, "raw_response": spec_result.raw_response},
                    latency_ms=spec_result.latency_ms,
                    status="error",
                )
            )
            return AgentRunResult(
                status="FAILED",
                up_sql="",
                down_sql="",
                spec=None,
                steps=steps,
                total_tokens=total_tokens,
                total_latency_ms=(time.time() - start_time) * 1000,
                stop_reason="PLAN_FAILED",
                notes=f"Failed to generate MigrationSpec: {spec_result.error}",
                fast_model=fast_model,
                ultra_model=ultra_model,
                model_warnings=warnings,
            )

    base_snapshot = adapter.clone()
    tool_executor = ToolExecutor(adapter, spec=spec, target_table=target_table, base_snapshot=base_snapshot)

    # Phase 2: Agent Tool/Draft Loop
    settings = get_settings(reload=True)
    wall_clock_budget = wall_clock_timeout_s if wall_clock_timeout_s is not None else settings.agent_wall_clock_timeout_s
    max_tool_rounds = settings.max_tool_rounds_per_step

    current_model = fast_model
    current_up_sql = ""
    current_down_sql = ""
    notes = ""
    inv_result: InvariantResult | None = None
    step_num = 0
    attempt_num = 1
    max_attempts = max_steps if max_steps is not None else getattr(settings, "max_attempts", 6)
    attempts: list[AttemptRecord] = []
    consecutive_no_sql = 0
    last_tool_sql = ""
    last_tool_down_sql = ""
    last_tool_result: InvariantResult | None = None
    has_drafted_sql = False
    repeat_feedback: str | None = None
    seen_sql_attempts: dict[str, tuple[int, list[str]]] = {}
    repeat_counts: dict[str, int] = {}
    failed_check_counts: dict[str, int] = {}
    failed_attempts_count = 0
    attempt_history = AttemptHistory()
    step_split_rule = ""

    # Put static schema, transform contract, and execution gotchas in system message once
    system_content = (
        "You are an expert database migration engineer writing SQLite migration SQL.\n\n"
        f"Current Database Schema:\n{format_schema_summary(schema)}\n\n"
        f"Approved Migration Plan (MigrationSpec):\n{spec.model_dump_json(indent=2) if spec else ''}\n\n"
        f"{generate_transform_contract(spec) if spec else ''}\n\n"
        f"{get_sqlite_gotchas_block()}\n\n"
        "Format Requirement: Prefer calling the submit(up_sql, down_sql) tool with your forward and rollback migrations explicitly separated. "
        "If returning markdown code blocks, you MUST use '-- ==== UP ====' for the forward migration and '-- ==== DOWN ====' for the rollback migration."
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]

    verified_sql_hash: str | None = None
    best_up_sql: str = ""
    best_down_sql: str = ""
    best_inv_result: InvariantResult | None = None
    best_failed_count: int = 999999
    best_passed_count: int = -1
    best_is_non_crash: bool = False

    while attempt_num <= max_attempts:
        step_num += 1
        step_start_time = time.time()
        elapsed_total = time.time() - start_time
        if elapsed_total > wall_clock_budget:
            final_up = best_up_sql or current_up_sql or (last_tool_sql if has_ddl_or_dml(last_tool_sql) else "")
            final_down = best_down_sql or current_down_sql or (last_tool_down_sql if last_tool_down_sql else "")
            return AgentRunResult(
                status="FAILED",
                up_sql=final_up,
                down_sql=final_down,
                spec=spec,
                steps=steps,
                attempts=attempts,
                invariant_result=best_inv_result or inv_result,
                total_tokens=total_tokens,
                total_latency_ms=elapsed_total * 1000,
                stop_reason=f"WALL_CLOCK_TIMEOUT: Exceeded {round(wall_clock_budget)}s budget.",
                fast_model=fast_model,
                ultra_model=ultra_model,
                model_warnings=warnings,
            )

        if repeat_feedback:
            notify(f"Re-prompting {current_model} after identical attempt detected...", {"step": step_num})
            traps = inv_result.trap_audit if inv_result else []
            sql_to_repair = best_up_sql or current_up_sql or last_tool_sql
            base_p = build_repair_prompt(
                schema,
                spec,
                sql_to_repair,
                inv_result,
                trap_results=traps,
                max_tool_calls=max_tool_rounds,
                history=attempt_history,
                split_rule=step_split_rule,
                down_sql=current_down_sql,
                target_table=target_table,
            )
            prompt = f"Your previous reply was {repeat_feedback}.\n\n{base_p}"
            repeat_feedback = None
            messages.append({"role": "user", "content": prompt})
        elif consecutive_no_sql > 0 and messages:
            notify(f"No SQL in reply, continuing same conversation (retry {consecutive_no_sql}/2)", {"step": step_num, "retry": consecutive_no_sql})
            messages.append({
                "role": "user",
                "content": (
                    "Your previous reply contained no executable migration SQL. "
                    "Reply now with the final migration in ```sql blocks, UP then '-- DOWN'."
                ),
            })
        elif inv_result is None:
            notify(f"Drafting initial migration using {current_model}...", {"step": step_num})
            prompt = build_migration_prompt(
                schema,
                spec,
                is_naive_baseline=is_naive_baseline,
                max_tool_calls=max_tool_rounds,
                profiles=profiles,
            )
            messages.append({"role": "user", "content": prompt})
        else:
            model_label = "Ultra" if "ultra" in current_model.lower() else ("Super" if "super" in current_model.lower() else current_model.split("/")[-1])
            first_fail_str = (inv_result.failed_checks[0][:160]) if (inv_result and inv_result.failed_checks) else (inv_result.error or "verification failure")
            notify(f"Attempt {attempt_num}/{max_attempts} ({model_label}): repairing {first_fail_str}", {"step": step_num, "attempt": attempt_num})
            traps = inv_result.trap_audit if inv_result else []
            # If an attempt crashes in the sandbox, repair from the best executable SQL plus crash message
            sql_to_repair = best_up_sql if (best_up_sql and inv_result and not inv_result.passed and inv_result.error) else (current_up_sql or last_tool_sql)
            prompt = build_repair_prompt(
                schema,
                spec,
                sql_to_repair,
                inv_result,
                trap_results=traps,
                max_tool_calls=max_tool_rounds,
                history=attempt_history,
                split_rule=step_split_rule,
                down_sql=current_down_sql,
                target_table=target_table,
            )
            messages.append({"role": "user", "content": prompt})

        step_up_sql = ""
        step_down_sql = ""
        tool_rounds = 0
        content = ""
        step_usage = None
        last_step_lat = 0.0

        for _ in range(max_tool_rounds):
            remaining_total = wall_clock_budget - (time.time() - start_time)
            if remaining_total < 5.0:
                logger.warning(
                    f"Total budget ({remaining_total:.1f}s) nearly exhausted. "
                    "Stopping tool loop to finalize migration."
                )
                break
            model_timeout = get_model_timeout(current_model, settings)
            call_timeout = min(remaining_total, model_timeout)
            t_call_start = time.time()

            tools_to_pass = TOOL_DEFINITIONS if consecutive_no_sql == 0 else None
            try:
                raw_msg, usage = call_llm(
                    client,
                    current_model,
                    messages=truncate_older_tool_results(messages),
                    temperature=0.2,
                    tools=tools_to_pass,
                    timeout=call_timeout,
                )
            except AgentError as ae:
                ae_msg = str(ae)
                is_timeout_or_transient = any(
                    k in ae_msg.lower()
                    for k in ("timed out", "timeout", "high load", "rate limit", "temporarily", "connection")
                )
                can_fallback_to_super = (
                    current_model == ultra_model
                    and ultra_model != fast_model
                    and is_timeout_or_transient
                    and attempt_num <= max_attempts
                    and (wall_clock_budget - (time.time() - start_time)) > 5.0
                )

                raw_msg = None
                if can_fallback_to_super:
                    fallback_msg = "Ultra model timed out; falling back to Super model for repair..."
                    logger.warning(fallback_msg)
                    notify(fallback_msg, {"step": step_num, "attempt": attempt_num, "error": ae_msg})
                    current_model = fast_model
                    super_timeout = get_model_timeout(current_model, settings)
                    rem_tot = wall_clock_budget - (time.time() - start_time)
                    retry_call_timeout = min(rem_tot, super_timeout)
                    try:
                        raw_msg, usage = call_llm(
                            client,
                            current_model,
                            messages=truncate_older_tool_results(messages),
                            temperature=0.2,
                            tools=tools_to_pass,
                            timeout=retry_call_timeout,
                        )
                    except AgentError as fallback_ae:
                        logger.error(f"Fallback to Super model failed: {fallback_ae}")
                        ae = fallback_ae

                if raw_msg is None:
                    logger.error(f"Agent LLM call failed: {ae}")
                    steps.append(
                        AgentStepLog(
                            step=len(steps) + 1,
                            model=current_model,
                            action="call_llm_error",
                            details={"error": str(ae)},
                            latency_ms=(time.time() - t_call_start) * 1000,
                            status="error",
                        )
                    )
                    final_up = best_up_sql or current_up_sql or (last_tool_sql if has_ddl_or_dml(last_tool_sql) else "")
                    final_down = best_down_sql or current_down_sql or (last_tool_down_sql if last_tool_down_sql else "")
                    return AgentRunResult(
                        status="FAILED",
                        up_sql=final_up,
                        down_sql=final_down,
                        spec=spec,
                        steps=steps,
                        attempts=attempts,
                        invariant_result=best_inv_result or inv_result,
                        total_tokens=total_tokens,
                        total_latency_ms=(time.time() - start_time) * 1000,
                        stop_reason=f"AGENT_ERROR: {ae}",
                        notes=f"Agent LLM error: {ae}",
                        fast_model=fast_model,
                        ultra_model=ultra_model,
                        model_warnings=warnings,
                    )

            call_lat = (time.time() - t_call_start) * 1000
            last_step_lat = call_lat
            step_usage = usage
            call_tokens = getattr(usage, "total_tokens", 0) if usage else 0
            total_tokens += call_tokens

            content = raw_msg.content or ""
            tool_calls = getattr(raw_msg, "tool_calls", None)

            if content:
                extracted_up, extracted_down, split_rule = parse_sql_scripts_ex(content)
                if extracted_up:
                    step_up_sql = extracted_up
                    step_down_sql = extracted_down
                    step_split_rule = split_rule

            if tool_calls:
                tool_rounds += 1
                tc_dicts = []
                for tc in tool_calls:
                    tc_id = getattr(tc, "id", "")
                    tc_type = getattr(tc, "type", "function")
                    fn_obj = getattr(tc, "function", None)
                    fn_name = getattr(fn_obj, "name", "")
                    fn_args = getattr(fn_obj, "arguments", "")
                    if not isinstance(fn_args, str):
                        fn_args = json.dumps(fn_args)
                    tc_dicts.append({
                        "id": tc_id,
                        "type": tc_type,
                        "function": {"name": fn_name, "arguments": fn_args},
                    })

                assistant_turn: dict[str, Any] = {"role": "assistant"}
                if content:
                    assistant_turn["content"] = strip_thinking(content)
                assistant_turn["tool_calls"] = tc_dicts
                messages.append(assistant_turn)

                for tc_dict in tc_dicts:
                    fn_name = tc_dict["function"]["name"]
                    fn_args = tc_dict["function"]["arguments"]
                    t_tool_start = time.time()
                    res = tool_executor.execute(fn_name, fn_args)
                    tool_lat = (time.time() - t_tool_start) * 1000

                    steps.append(
                        AgentStepLog(
                            step=len(steps) + 1,
                            model=current_model,
                            action=f"tool:{fn_name}",
                            details={
                                "raw_reply": content[:500] if content else "",
                                "args": fn_args,
                                "result": res,
                            },
                            tokens_used=0,
                            latency_ms=tool_lat,
                            status="ok" if "error" not in res else "error",
                        )
                    )

                    if fn_name == "submit":
                        step_up_sql = res.get("up_sql", "")
                        step_down_sql = res.get("down_sql", "")
                        notes = res.get("notes", "")
                    elif fn_name == "run_on_clone":
                        try:
                            p_args = json.loads(fn_args) if isinstance(fn_args, str) else fn_args
                            candidate_sql = p_args.get("up_sql", "") if isinstance(p_args, dict) else ""
                            if candidate_sql and has_ddl_or_dml(candidate_sql):
                                last_tool_sql = candidate_sql
                                if "down_sql" in p_args and p_args["down_sql"]:
                                    last_tool_down_sql = p_args["down_sql"]
                                    if not step_down_sql:
                                        step_down_sql = p_args["down_sql"]
                                if res.get("status") in ("VERIFIED", "UNVERIFIED", "FAILED", "PASSED"):
                                    last_tool_result = InvariantResult.model_validate(res)
                                    logger.debug(f"Tool run_on_clone candidate result: {last_tool_result.status}")
                        except Exception:
                            pass

                    res_str = json.dumps(res)
                    if len(res_str) > 4000:
                        res_str = res_str[:4000] + "... [truncated]"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_dict["id"],
                        "content": res_str,
                    })

                if step_up_sql:
                    break
            else:
                messages.append({"role": "assistant", "content": strip_thinking(content)})
                break

        # Final nudge call: if tool rounds budget reached and no SQL produced yet
        remaining_total = wall_clock_budget - (time.time() - start_time)
        model_timeout = get_model_timeout(current_model, settings)
        call_timeout = min(remaining_total, model_timeout)
        if not step_up_sql and remaining_total >= 10.0 and messages and messages[-1].get("role") == "tool":
            t_nudge_start = time.time()
            final_reminder = {
                "role": "user",
                "content": "Reply now with the final migration in ```sql blocks, UP then '-- DOWN'.",
            }
            messages.append(final_reminder)
            try:
                raw_msg, usage = call_llm(
                    client,
                    current_model,
                    messages=messages,
                    temperature=0.2,
                    tools=None,
                    timeout=call_timeout,
                )
                nudge_lat = (time.time() - t_nudge_start) * 1000
                nudge_tokens = getattr(usage, "total_tokens", 0) if usage else 0
                total_tokens += nudge_tokens

                content = raw_msg.content or ""
                messages.append({"role": "assistant", "content": strip_thinking(content)})
                if content:
                    extracted_up, extracted_down, split_rule = parse_sql_scripts_ex(content)
                    if extracted_up:
                        step_up_sql = extracted_up
                        step_down_sql = extracted_down
                        step_split_rule = split_rule

                steps.append(
                    AgentStepLog(
                        step=len(steps) + 1,
                        model=current_model,
                        action="final_nudge",
                        details={"raw_reply": content[:500] if content else ""},
                        tokens_used=nudge_tokens,
                        latency_ms=nudge_lat,
                        status="ok",
                    )
                )
            except AgentError as ae:
                logger.warning(f"Final tool completion call failed: {ae}")

        # Handle No-SQL fallback or retry
        if not step_up_sql:
            if last_tool_sql and has_ddl_or_dml(last_tool_sql):
                logger.info("Using candidate SQL from tool execution as fallback immediately on no-SQL reply.")
                step_up_sql = last_tool_sql
                if not step_down_sql and last_tool_down_sql:
                    step_down_sql = last_tool_down_sql
                consecutive_no_sql = 0
            else:
                consecutive_no_sql += 1
                steps.append(
                    AgentStepLog(
                        step=len(steps) + 1,
                        model=current_model,
                        action="no_sql_reply",
                        details={"raw_reply": content[:500] if content else ""},
                        tokens_used=getattr(step_usage, "total_tokens", 0) if step_usage else 0,
                        latency_ms=last_step_lat,
                        status="warning",
                    )
                )
                if consecutive_no_sql >= 2:
                    final_up = best_up_sql or current_up_sql
                    final_down = best_down_sql or current_down_sql
                    return AgentRunResult(
                        status="FAILED",
                        up_sql=final_up,
                        down_sql=final_down,
                        spec=spec,
                        steps=steps,
                        attempts=attempts,
                        invariant_result=best_inv_result or inv_result,
                        total_tokens=total_tokens,
                        total_latency_ms=(time.time() - start_time) * 1000,
                        stop_reason="MODEL_RETURNED_NO_SQL",
                        fast_model=fast_model,
                        ultra_model=ultra_model,
                        model_warnings=warnings,
                    )
                logger.warning(f"Model produced empty SQL; retrying conversation (retry {consecutive_no_sql}/2)...")
                continue

        no_sql_replies_count = consecutive_no_sql
        consecutive_no_sql = 0
        last_tool_sql = ""
        last_tool_down_sql = ""

        # Check for repeated SQL / no-progress
        norm_sql = normalize_sql_for_progress(step_up_sql)
        sql_hash = hashlib.sha256(norm_sql.encode()).hexdigest()

        if sql_hash in seen_sql_attempts:
            orig_step, orig_failed = seen_sql_attempts[sql_hash]
            repeat_counts[sql_hash] = repeat_counts.get(sql_hash, 0) + 1
            if repeat_counts[sql_hash] >= 2:
                steps.append(
                    AgentStepLog(
                        step=len(steps) + 1,
                        model=current_model,
                        action="no_progress",
                        details={"repeat_of_step": orig_step, "sql_preview": step_up_sql[:200]},
                        status="error",
                    )
                )
                final_up = best_up_sql or current_up_sql
                final_down = best_down_sql or current_down_sql
                return AgentRunResult(
                    status="FAILED",
                    up_sql=final_up,
                    down_sql=final_down,
                    spec=spec,
                    steps=steps,
                    attempts=attempts,
                    invariant_result=best_inv_result or inv_result,
                    total_tokens=total_tokens,
                    total_latency_ms=(time.time() - start_time) * 1000,
                    stop_reason="NO_PROGRESS",
                    notes=f"Model returned identical SQL to attempt {orig_step} twice without progress.",
                    fast_model=fast_model,
                    ultra_model=ultra_model,
                    model_warnings=warnings,
                )
            else:
                failed_str = ", ".join(orig_failed) if orig_failed else "verification failure"
                repeat_feedback = f"identical to attempt {orig_step} which failed with {failed_str}; change approach"
                steps.append(
                    AgentStepLog(
                        step=len(steps) + 1,
                        model=current_model,
                        action="repeat_sql_skipped",
                        details={"repeat_of_step": orig_step, "feedback": repeat_feedback},
                        status="warning",
                    )
                )
                current_model = ultra_model
                continue

        current_up_sql = step_up_sql
        if step_down_sql:
            current_down_sql = step_down_sql

        action_label = "draft_migration" if not has_drafted_sql else "repair_migration"
        has_drafted_sql = True

        steps.append(
            AgentStepLog(
                step=len(steps) + 1,
                model=current_model,
                action=action_label,
                details={
                    "raw_reply": content[:500] if content else "",
                    "up_sql_preview": step_up_sql[:200],
                },
                tokens_used=getattr(step_usage, "total_tokens", 0) if step_usage else 0,
                latency_ms=last_step_lat,
                status="ok",
            )
        )

        # Dry-run on fresh clone from base snapshot
        notify("Executing dry-run verification on disposable sandbox clone...", {"step": step_num})
        base_clone = sqlite3.connect(":memory:", check_same_thread=False)
        base_snapshot.backup(base_clone)
        test_clone = sqlite3.connect(":memory:", check_same_thread=False)
        base_snapshot.backup(test_clone)
        real_before = 0
        if target_table:
            try:
                cur = base_clone.cursor()
                cur.execute(f'SELECT COUNT(*) FROM "{target_table}"')
                row = cur.fetchone()
                if row:
                    real_before = int(row[0])
            except Exception:
                pass

        try:
            run_ok, run_err, run_elapsed = adapter.run_in_sandbox(test_clone, current_up_sql, fk_enabled=True)

            if not run_ok:
                err_msg = run_err or "Execution error"
                hint = ""
                m_nn = re.search(r"NOT NULL constraint failed:\s*([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", err_msg)
                if m_nn:
                    col_name = m_nn.group(2)
                    try:
                        cur_b = base_clone.cursor()
                        cur_b.execute(f'SELECT COUNT(*) FROM "{target_table}" WHERE "{col_name}" IS NULL')
                        null_cnt = cur_b.fetchone()[0]
                    except Exception:
                        null_cnt = "?"
                    mapping = next((m for m in spec.get_table_mappings(target_table) if m.new_column == col_name), None) if spec else None
                    if mapping and mapping.default_value is not None:
                        hint = f" [How to fix: Column '{col_name}' has {null_cnt} NULL rows in source table. Use COALESCE({col_name}, '{mapping.default_value}') to fill defaults as specified in spec.]"
                    else:
                        hint = f" [How to fix: Column '{col_name}' has {null_cnt} NULL rows in source table. Keep '{col_name}' nullable in the table definition or provide a safe default.]"
                multi_diag = audit_crash_diagnostics(
                    base_clone, target_table, err_msg, current_sql=current_up_sql, spec=spec
                )
                full_err = f"{err_msg}{multi_diag or hint}"
                inv_result = InvariantResult(
                    status="FAILED",
                    passed=False,
                    error=full_err,
                    failed_checks=[full_err],
                    elapsed_ms=run_elapsed,
                    before_count=real_before,
                )
            else:
                inv_result = evaluate_migration_invariants(
                    base_clone,
                    test_clone,
                    target_table=target_table,
                    spec=spec,
                    elapsed_ms=run_elapsed,
                )
                try:
                    traps = audit_traps(test_clone, target_table)
                    inv_result.trap_audit = traps
                    failing_traps = [t for t in traps if not t.get("survived")]
                    if failing_traps:
                        inv_result.status = "FAILED"
                        inv_result.passed = False
                        for ft in failing_traps:
                            inv_result.failed_checks.append(
                                f"ADVERSARIAL_TRAP_FAILED: Trap '{ft.get('trap_type')}' failed ({ft.get('description')})"
                            )
                except Exception as exc:
                    logger.debug(f"No trap audit available: {exc}")

                # Self-undo guard: after UP runs, if target table columns == baseline columns while spec expects new columns,
                # or UP contains statements that recreate the baseline table name from a rollback pattern
                if not inv_result.passed and spec:
                    cur_m = test_clone.cursor()
                    cur_m.execute(f'PRAGMA table_info("{target_table}")')
                    test_cols = [r[1] for r in cur_m.fetchall()]
                    cur_b = base_clone.cursor()
                    cur_b.execute(f'PRAGMA table_info("{target_table}")')
                    base_cols = [r[1] for r in cur_b.fetchall()]

                    spec_req_cols = spec.get_required_columns(target_table)
                    expects_new_columns = any(c not in base_cols for c in spec_req_cols)
                    has_rollback_rename = bool(
                        re.search(r"(?i)\b(?:RENAME\s+TO\s+" + target_table + r")\b", current_up_sql)
                        and ("_old" in current_up_sql or "_prev" in current_up_sql or "_orig" in current_up_sql)
                    )

                    if (test_cols == base_cols and expects_new_columns) or (
                        has_rollback_rename and any("MISSING_REQUIRED_COLUMNS" in fc for fc in inv_result.failed_checks)
                    ):
                        logger.warning("Detected self-undo pattern in up_sql! Re-splitting using last block = DOWN...")
                        re_blocks = list(re.finditer(r"(?:^|\n)```([^\n]*)\n(.*?)\n```", content or "", re.DOTALL))
                        if len(re_blocks) >= 2:
                            re_up = "\n\n".join(_clean_extracted_block(m.group(2)) for m in re_blocks[:-1] if m.group(2).strip())
                            re_down = _clean_extracted_block(re_blocks[-1].group(2))
                            retry_clone = sqlite3.connect(":memory:", check_same_thread=False)
                            base_snapshot.backup(retry_clone)
                            try:
                                re_ok, re_err, re_el = adapter.run_in_sandbox(retry_clone, re_up, fk_enabled=True)
                                if re_ok:
                                    re_inv = evaluate_migration_invariants(
                                        base_clone, retry_clone, target_table=target_table, spec=spec, elapsed_ms=re_el
                                    )
                                    if re_inv.passed:
                                        logger.info("Re-split with last block as DOWN succeeded and passed verification!")
                                        current_up_sql = re_up
                                        current_down_sql = re_down
                                        inv_result = re_inv
                            finally:
                                retry_clone.close()

                        if not inv_result.passed:
                            inv_result.failed_checks.insert(
                                0, "UP_UNDONE: your up_sql contains a rollback section; keep UP and DOWN separate"
                            )
        finally:
            base_clone.close()
            test_clone.close()

        first_fail = (inv_result.failed_checks[0][:160]) if inv_result.failed_checks else ""
        fail_suffix = f" — {first_fail}" if (inv_result.status == "FAILED" and first_fail) else ""
        notify(
            f"Verification outcome: {inv_result.status} (passed {len(inv_result.passed_checks)}, failed {len(inv_result.failed_checks)}){fail_suffix}",
            {"status": inv_result.status, "inv_result": inv_result.model_dump()},
        )

        attempt_history.record_attempt(
            attempt_num=attempt_num,
            model=current_model,
            up_sql=current_up_sql,
            inv_result=inv_result,
            down_sql=current_down_sql,
            split_rule=step_split_rule,
        )

        attempts.append(
            AttemptRecord(
                attempt=attempt_num,
                model=current_model,
                passed_count=len(inv_result.passed_checks),
                failed_count=len(inv_result.failed_checks),
                first_failed_check=first_fail,
                sandbox_error=run_err if not run_ok else None,
                latency_ms=(time.time() - step_start_time) * 1000,
                tokens=getattr(step_usage, "total_tokens", 0) if step_usage else 0,
                raw_reply=(content or "")[:1000],
                up_sql=current_up_sql,
                down_sql=current_down_sql,
                tool_rounds_used=tool_rounds,
                no_sql_replies=no_sql_replies_count,
                split_rule=step_split_rule,
            )
        )

        cleaned_content = strip_thinking(content)
        if not any(m.get("role") == "assistant" and (m.get("content") or m.get("tool_calls")) for m in messages[-2:]):
            messages.append({"role": "assistant", "content": cleaned_content or f"```sql\n{current_up_sql}\n```"})

        # Best-attempt tracking:
        # Non-crashes (run_ok is True and passed_count > 0) strictly outrank crashes (run_ok is False)
        # Ties are broken by fewest failed checks, then most passed checks.
        failed_count = len(inv_result.failed_checks)
        passed_count = len(inv_result.passed_checks)
        is_non_crash = (run_ok is True and passed_count > 0)

        is_better = False
        if is_non_crash and not best_is_non_crash:
            is_better = True
        elif is_non_crash == best_is_non_crash:
            if failed_count < best_failed_count:
                is_better = True
            elif failed_count == best_failed_count and passed_count > best_passed_count:
                is_better = True

        if is_better:
            best_failed_count = failed_count
            best_passed_count = passed_count
            best_is_non_crash = is_non_crash
            best_up_sql = current_up_sql
            best_down_sql = current_down_sql
            best_inv_result = inv_result

        if inv_result.passed and inv_result.status in ("VERIFIED", "UNVERIFIED"):
            verified_sql_hash = hashlib.sha256(normalize_sql_for_progress(current_up_sql).encode()).hexdigest()
            break

        failed_attempts_count += 1
        seen_sql_attempts[sql_hash] = (attempt_num, list(inv_result.failed_checks))
        attempt_num += 1

        # Check for repeated failed checks
        repeat_check = False
        for fc in inv_result.failed_checks:
            failed_check_counts[fc] = failed_check_counts.get(fc, 0) + 1
            if failed_check_counts[fc] >= 2:
                repeat_check = True

        # Escalation policy:
        # stay on Super for sandbox/syntax errors and first data failure;
        # escalate to Ultra when the same failed check repeats twice or after 2 failed attempts.
        if repeat_check or failed_attempts_count >= 2:
            current_model = ultra_model
        else:
            current_model = fast_model

    # Phase 3: Rollback Verification & Roundtrip Test
    # If verification passed, use verified SQL; if not, return the best attempt
    is_forward_verified = bool(inv_result and inv_result.passed and inv_result.status in ("VERIFIED", "UNVERIFIED"))
    if is_forward_verified:
        final_up = current_up_sql
        final_down = current_down_sql
    else:
        final_up = best_up_sql or current_up_sql
        final_down = best_down_sql or current_down_sql
        if best_inv_result is not None:
            inv_result = best_inv_result
            if inv_result.passed and inv_result.status in ("VERIFIED", "UNVERIFIED"):
                is_forward_verified = True

    rollback_ok = False
    rollback_err: str | None = None
    lossy_cols: list[str] = []
    rb_res = None
    if is_forward_verified and final_up and final_down:
        notify("Verifying rollback roundtrip on disposable clone...")
        base_clone = sqlite3.connect(":memory:", check_same_thread=False)
        base_snapshot.backup(base_clone)
        try:
            # Query baseline table DDL and nullable columns
            cur_snap = base_snapshot.cursor()
            cur_snap.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (target_table,))
            snap_row = cur_snap.fetchone()
            baseline_table_ddl = snap_row[0] if snap_row else ""
            cur_snap.execute(f'PRAGMA table_info("{target_table}")')
            b_cols_meta = cur_snap.fetchall()
            nullable_cols = [r[1] for r in b_cols_meta if not r[3] and r[5] == 0]

            final_down = sanitize_down_script(final_down, nullable_cols)

            rb_res = verify_rollback_roundtrip(
                base_clone,
                final_up,
                final_down,
                target_table=target_table,
                fk_enabled=True,
                base_timeout_s=settings.sandbox_timeout_s,
            )
            rollback_ok = rb_res.rollback_verified
            rollback_err = rb_res.error
            lossy_cols = rb_res.lossy_columns

            # If rollback is not verified (BROKEN or LOSSY), run repair step for DOWN script,
            # but only if the error was not a failure inside the UP migration
            is_forward_crash = bool(rb_res.error and "FORWARD_MIGRATION" in rb_res.error)
            rb_tries = 0
            while (not rollback_ok) and (not is_forward_crash) and rb_tries < 2:
                remaining_budget = max(0.0, wall_clock_budget - (time.time() - start_time))
                if remaining_budget < 10.0:
                    break
                rb_tries += 1
                notify(f"Rollback roundtrip {rb_res.classification}: repairing DOWN script (attempt {rb_tries}/2)...")
                diffs_text = (
                    "; ".join([f"PK {d['pk']} col {d['column']}: expected {d['original']!r}, got {d['restored']!r}" for d in rb_res.sample_diffs])
                    if rb_res.sample_diffs
                    else "None"
                )
                down_repair_prompt = f"""The forward migration (UP) succeeded and is verified:
```sql
{final_up}
```

However, the rollback script (DOWN) failed verification:
Classification: {rb_res.classification}
Error: {rb_res.error}
Missing/Lossy columns: {rb_res.lossy_columns}
Sample diffs: {diffs_text}

Pre-Migration Baseline Schema for '{target_table}':
```sql
{baseline_table_ddl}
```
Baseline Nullable Columns: {nullable_cols}

CRITICAL RULES FOR DOWN (ROLLBACK) MIGRATION:
1. The DOWN migration must restore table '{target_table}' to its original baseline schema above.
2. In your table rebuild (e.g. CREATE TABLE {target_table}_restore ...), preserve the EXACT column nullability from the baseline schema above. NEVER add NOT NULL to columns that were nullable in the baseline: {nullable_cols}.
3. If the UP migration inserted sentinel rows into referenced parent tables (e.g. 'UNASSIGNED'), clean them up:
   DELETE FROM parent_table WHERE pk = 'UNASSIGNED';
4. Do NOT mutate or alter untouched data values during rollback.

Task: Provide ONLY the corrected DOWN migration inside a ```sql block to restore table '{target_table}' to its original baseline state."""

                try:
                    call_to = min(remaining_budget, get_model_timeout(ultra_model, settings))
                    raw_msg, _ = call_llm(
                        client,
                        ultra_model,
                        messages=[{"role": "user", "content": down_repair_prompt}],
                        temperature=0.1,
                        tools=None,
                        timeout=call_to,
                    )
                    content = raw_msg.content or ""
                    _, new_down = parse_sql_scripts(content)
                    if not new_down:
                        new_down = clean_sql(content)
                    if new_down and has_ddl_or_dml(new_down):
                        final_down = sanitize_down_script(new_down, nullable_cols)
                        rb_res = verify_rollback_roundtrip(
                            base_clone,
                            final_up,
                            final_down,
                            target_table=target_table,
                            fk_enabled=True,
                            base_timeout_s=settings.sandbox_timeout_s,
                        )
                        rollback_ok = rb_res.rollback_verified
                        rollback_err = rb_res.error
                        lossy_cols = rb_res.lossy_columns
                        if rollback_ok:
                            notify("Rollback roundtrip repair succeeded!")
                            break
                except Exception as exc:
                    logger.debug(f"Rollback DOWN repair call failed: {exc}")
                    break
        finally:
            base_clone.close()

        if not rollback_ok and rb_res:
            if rb_res.classification == "LOSSY":
                notify(f"Rollback warning (LOSSY): {rollback_err}", {"lossy_columns": lossy_cols, "sample_diffs": rb_res.sample_diffs})
            else:
                notify(f"Rollback error ({rb_res.classification}): {rollback_err}", {"lossy_columns": lossy_cols})
    elif not is_forward_verified:
        rollback_err = "Forward migration did not pass dry-run verification; rollback verification skipped."
    elif not final_down:
        rollback_err = "no valid rollback script provided"

    # Estimate lock time
    lock_est = estimate_production_lock_time(
        inv_result.elapsed_ms if inv_result else 0.0,
        clone_rows=inv_result.before_count if inv_result else 0,
        production_rows=prod_rows,
    )

    final_status = "FAILED"
    if inv_result and inv_result.passed:
        final_hash = hashlib.sha256(normalize_sql_for_progress(final_up).encode()).hexdigest() if final_up else None
        if verified_sql_hash and final_hash == verified_sql_hash:
            final_status = inv_result.status  # VERIFIED or UNVERIFIED
        else:
            final_status = "FAILED"

    stop_reason = (
        "SUCCESS: All verification invariants passed."
        if final_status in ("VERIFIED", "UNVERIFIED")
        else f"MAX_ATTEMPTS_EXHAUSTED (attempts used {len(attempts)}/{max_attempts})"
    )

    total_lat = (time.time() - start_time) * 1000
    try:
        base_snapshot.close()
    except Exception:
        pass

    return AgentRunResult(
        status=final_status,
        up_sql=final_up,
        down_sql=final_down,
        spec=spec,
        steps=steps,
        attempts=attempts,
        invariant_result=inv_result,
        rollback_verified=rollback_ok,
        rollback_error=rollback_err,
        lossy_columns=lossy_cols,
        rollback_classification=rb_res.classification if rb_res else "",
        lock_estimate=lock_est,
        total_tokens=total_tokens,
        total_latency_ms=total_lat,
        stop_reason=stop_reason,
        notes=notes,
        fast_model=fast_model,
        ultra_model=ultra_model,
        model_warnings=warnings,
    )

