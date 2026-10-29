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
from timemachine.agent.planner import generate_spec, validate_spec_against_schema
from timemachine.agent.prompts import (
    build_migration_prompt,
    build_repair_prompt,
)
from timemachine.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from timemachine.config import AgentError, get_settings
from timemachine.logging import get_logger
from timemachine.traps.generator import audit_traps
from timemachine.traps.profiler import profile_database
from timemachine.verify.invariants import InvariantResult, evaluate_migration_invariants
from timemachine.verify.rebuild import estimate_production_lock_time, verify_rollback_roundtrip
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


def parse_sql_scripts(content: str) -> tuple[str, str]:
    """Parse UP and optional DOWN migration scripts from LLM content without guessing by position."""
    text = content.replace("\r\n", "\n")

    # 1. Check for fenced blocks first
    fence_pattern = re.compile(
        r"(?:^|\n)```([^\n]*)\n(.*?)\n```",
        re.DOTALL,
    )
    matches = list(fence_pattern.finditer(text))
    if matches:
        down_start_idx = None
        for i, m in enumerate(matches):
            tag = m.group(1).lower()
            block_body = m.group(2).strip().lower()
            if "down" in tag or "rollback" in tag:
                down_start_idx = i
                break
            if block_body.startswith(("-- down", "--down", "-- rollback", "--rollback")):
                down_start_idx = i
                break
            prev_end = matches[i - 1].end() if i > 0 else 0
            prose = text[prev_end:m.start()].strip()
            if re.search(r"(?im)\b(?:down|rollback)\b", prose):
                down_start_idx = i
                break

        if down_start_idx is not None and down_start_idx > 0:
            up_blocks = [_clean_extracted_block(matches[j].group(2)) for j in range(down_start_idx) if matches[j].group(2).strip()]
            down_blocks = [_clean_extracted_block(matches[j].group(2)) for j in range(down_start_idx, len(matches)) if matches[j].group(2).strip()]
            up = "\n\n".join(b for b in up_blocks if b)
            down = "\n\n".join(b for b in down_blocks if b)
            if not has_ddl_or_dml(down):
                down = ""
            return up, down

    # 2. Split on a down/rollback line (either plain comment or fence opening with down/rollback comment)
    down_line_re = re.compile(r"(?im)^\s*(?:```[a-zA-Z0-9_-]*\s*)?--\s*(?:DOWN|ROLLBACK)\b.*$")
    match = down_line_re.search(text)
    if match:
        up_part = text[:match.start()]
        down_part = text[match.end():]
        up = _clean_sql_section(up_part)
        down = _clean_sql_section(down_part)
        if not has_ddl_or_dml(down):
            down = ""
        return up, down

    if matches:
        all_blocks = [_clean_extracted_block(m.group(2)) for m in matches if m.group(2).strip()]
        up = "\n\n".join(b for b in all_blocks if b)
        return up, ""

    return clean_sql(text), ""


def run_time_machine_agent(
    adapter: DbAdapter,
    goal: str,
    spec: MigrationSpec | None = None,
    target_table: str = "employees",
    api_key_override: str | None = None,
    max_steps: int | None = None,
    is_naive_baseline: bool = False,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
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

    # Phase 1: Planning (derive spec if not provided, or validate supplied spec)
    if spec is not None:
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
        clone = adapter.clone()
        try:
            profiles = profile_database(clone, schema)
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
    wall_clock_budget = settings.agent_wall_clock_timeout_s
    max_tool_rounds = settings.max_tool_rounds_per_step

    current_model = fast_model
    current_up_sql = ""
    current_down_sql = ""
    notes = ""
    inv_result: InvariantResult | None = None
    step_num = 0
    attempt_num = 1
    max_attempts = max_steps if max_steps is not None else getattr(settings, "max_attempts", 6)
    step_cap_s = wall_clock_budget / max(max_attempts, 1)
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
    messages: list[dict[str, Any]] = []

    verified_sql_hash: str | None = None
    best_up_sql: str = ""
    best_down_sql: str = ""
    best_inv_result: InvariantResult | None = None
    best_failed_count: int = 999999
    best_passed_count: int = -1

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
            base_p = build_repair_prompt(schema, spec, sql_to_repair, inv_result, trap_results=traps, max_tool_calls=max_tool_rounds)
            prompt = f"Your previous reply was {repeat_feedback}.\n\n{base_p}"
            repeat_feedback = None
            messages = [{"role": "user", "content": prompt}]
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
            prompt = build_migration_prompt(schema, spec, is_naive_baseline=is_naive_baseline, max_tool_calls=max_tool_rounds)
            messages = [{"role": "user", "content": prompt}]
        else:
            model_label = "Ultra" if "ultra" in current_model.lower() else ("Super" if "super" in current_model.lower() else current_model.split("/")[-1])
            first_fail_str = (inv_result.failed_checks[0][:160]) if (inv_result and inv_result.failed_checks) else (inv_result.error or "verification failure")
            notify(f"Attempt {attempt_num}/{max_attempts} ({model_label}): repairing {first_fail_str}", {"step": step_num, "attempt": attempt_num})
            traps = inv_result.trap_audit if inv_result else []
            # If an attempt crashes in the sandbox, repair from the best executable SQL plus crash message
            sql_to_repair = best_up_sql if (best_up_sql and inv_result and not inv_result.passed and inv_result.error) else (current_up_sql or last_tool_sql)
            prompt = build_repair_prompt(schema, spec, sql_to_repair, inv_result, trap_results=traps, max_tool_calls=max_tool_rounds)
            messages = [{"role": "user", "content": prompt}]

        step_up_sql = ""
        step_down_sql = ""
        tool_rounds = 0
        content = ""
        step_usage = None
        last_step_lat = 0.0

        for _ in range(max_tool_rounds):
            remaining_total = wall_clock_budget - (time.time() - start_time)
            remaining_step = step_cap_s - (time.time() - step_start_time)
            if remaining_total < 10.0 or remaining_step < 5.0:
                logger.warning(
                    f"Step ({remaining_step:.1f}s) or total budget ({remaining_total:.1f}s) nearly exhausted. "
                    "Stopping tool loop to finalize migration."
                )
                break
            call_timeout = min(remaining_total, remaining_step, settings.llm_timeout_s)
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
                extracted_up, extracted_down = parse_sql_scripts(content)
                if extracted_up:
                    step_up_sql = extracted_up
                    step_down_sql = extracted_down

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
                    assistant_turn["content"] = content
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
                messages.append({"role": "assistant", "content": content})
                break

        # Final nudge call: if tool rounds budget reached and no SQL produced yet
        remaining_total = wall_clock_budget - (time.time() - start_time)
        remaining_step = step_cap_s - (time.time() - step_start_time)
        call_timeout = min(remaining_total, remaining_step, settings.llm_timeout_s)
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
                messages.append({"role": "assistant", "content": content})
                if content:
                    extracted_up, extracted_down = parse_sql_scripts(content)
                    if extracted_up:
                        step_up_sql = extracted_up
                        step_down_sql = extracted_down

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
                inv_result = InvariantResult(
                    status="FAILED",
                    passed=False,
                    error=run_err,
                    failed_checks=[run_err or "Execution error"],
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
        finally:
            base_clone.close()
            test_clone.close()

        first_fail = (inv_result.failed_checks[0][:160]) if inv_result.failed_checks else ""
        fail_suffix = f" — {first_fail}" if (inv_result.status == "FAILED" and first_fail) else ""
        notify(
            f"Verification outcome: {inv_result.status} (passed {len(inv_result.passed_checks)}, failed {len(inv_result.failed_checks)}){fail_suffix}",
            {"status": inv_result.status, "inv_result": inv_result.model_dump()},
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
            )
        )

        # Best-attempt tracking: fewest failed checks (ties: most passed checks)
        failed_count = len(inv_result.failed_checks)
        passed_count = len(inv_result.passed_checks)
        if failed_count < best_failed_count or (failed_count == best_failed_count and passed_count > best_passed_count):
            best_failed_count = failed_count
            best_passed_count = passed_count
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

    rollback_ok = False
    rollback_err: str | None = None
    lossy_cols: list[str] = []
    rb_res = None
    if is_forward_verified and final_up and final_down:
        notify("Verifying rollback roundtrip on disposable clone...")
        base_clone = sqlite3.connect(":memory:", check_same_thread=False)
        base_snapshot.backup(base_clone)
        try:
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

Task: Provide ONLY the corrected DOWN migration inside a ```sql block to restore table '{target_table}' to its original baseline state."""

                try:
                    call_to = min(settings.llm_timeout_s, 30.0, max(2.0, remaining_budget))
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
                        final_down = new_down
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
        else "STEP_BUDGET_EXHAUSTED"
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

