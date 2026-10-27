"""Autonomous agent loop with tool-calling, tiered model escalation, and structured telemetry."""

import hashlib
import json
import re
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


class AgentRunResult(BaseModel):
    """Final outcome of an autonomous migration dry-run execution."""
    status: str  # VERIFIED, UNVERIFIED, FAILED
    up_sql: str = ""
    down_sql: str = ""
    spec: MigrationSpec | None = None
    steps: list[AgentStepLog] = Field(default_factory=list)
    invariant_result: InvariantResult | None = None
    rollback_verified: bool = False
    rollback_error: str | None = None
    lossy_columns: list[str] = Field(default_factory=list)
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


def parse_sql_scripts(content: str) -> tuple[str, str]:
    """Parse UP and optional DOWN migration scripts from LLM content without guessing by position."""
    text = content.replace("\r\n", "\n")

    # 1. Split on a down/rollback line (either plain comment or fence opening with down/rollback comment)
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

    # 2. Check for fenced blocks with down/rollback fence tags or preceding prose labelled down/rollback
    fence_pattern = re.compile(
        r"(?:^|\n)```([^\n]*)\n(.*?)\n```",
        re.DOTALL,
    )
    matches = list(fence_pattern.finditer(text))
    if matches:
        down_start_idx = None
        for i, m in enumerate(matches):
            tag = m.group(1).lower()
            if "down" in tag or "rollback" in tag:
                down_start_idx = i
                break
            prev_end = matches[i - 1].end() if i > 0 else 0
            prose = text[prev_end:m.start()].strip()
            if re.search(r"(?im)^\s*(?:down|rollback|down\s+migration|rollback\s+script):?\s*$", prose):
                down_start_idx = i
                break

        if down_start_idx is not None:
            up_blocks = [_clean_extracted_block(matches[j].group(2)) for j in range(down_start_idx) if matches[j].group(2).strip()]
            down_blocks = [_clean_extracted_block(matches[j].group(2)) for j in range(down_start_idx, len(matches)) if matches[j].group(2).strip()]
            up = "\n\n".join(b for b in up_blocks if b)
            down = "\n\n".join(b for b in down_blocks if b)
            if not has_ddl_or_dml(down):
                down = ""
            return up, down

        # Otherwise treat ALL fenced blocks as UP (concatenate in order) and down = ""
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
    max_steps: int = 8,
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

    tool_executor = ToolExecutor(adapter, spec=spec, target_table=target_table)

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
    consecutive_no_sql = 0
    last_raw_reply = ""
    last_tool_sql = ""
    last_tool_down_sql = ""
    has_drafted_sql = False
    repeat_feedback: str | None = None
    seen_sql_attempts: dict[str, tuple[int, list[str]]] = {}
    repeat_counts: dict[str, int] = {}
    messages: list[dict[str, Any]] = []

    while step_num < max_steps:
        step_num += 1
        elapsed_total = time.time() - start_time
        if elapsed_total > wall_clock_budget:
            final_up = current_up_sql or (last_tool_sql if has_ddl_or_dml(last_tool_sql) else "")
            final_down = current_down_sql or (last_tool_down_sql if last_tool_down_sql else "")
            return AgentRunResult(
                status="FAILED",
                up_sql=final_up,
                down_sql=final_down,
                spec=spec,
                steps=steps,
                invariant_result=inv_result,
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
            sql_to_repair = current_up_sql or last_tool_sql
            base_p = build_repair_prompt(schema, spec, sql_to_repair, inv_result, trap_results=traps)
            prompt = f"Your previous reply was {repeat_feedback}.\n\n{base_p}"
            repeat_feedback = None
            messages = [{"role": "user", "content": prompt}]
        elif inv_result is None:
            if consecutive_no_sql > 0 and messages:
                notify(f"Re-prompting {current_model} for migration SQL...", {"step": step_num})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous reply contained no executable migration SQL. "
                        "Based on the database inspection and tool outputs above, please provide "
                        "the complete SQLite migration script inside ```sql blocks with both UP and DOWN migrations, "
                        "or call submit."
                    ),
                })
            elif consecutive_no_sql > 0 and last_raw_reply:
                notify(f"Re-prompting {current_model} for migration SQL...", {"step": step_num})
                prompt = build_repair_prompt(schema, spec, last_raw_reply, None)
                messages = [{"role": "user", "content": prompt}]
            else:
                notify(f"Drafting initial migration using {current_model}...", {"step": step_num})
                prompt = build_migration_prompt(schema, spec, is_naive_baseline=is_naive_baseline)
                messages = [{"role": "user", "content": prompt}]
        else:
            notify(f"Escalating repair to {current_model} with structured failure telemetry...", {"step": step_num})
            traps = inv_result.trap_audit if inv_result else []
            sql_to_repair = current_up_sql or last_tool_sql
            prompt = build_repair_prompt(schema, spec, sql_to_repair, inv_result, trap_results=traps)
            messages = [{"role": "user", "content": prompt}]

        step_up_sql = ""
        step_down_sql = ""
        tool_rounds = 0
        content = ""
        last_step_lat = 0.0
        step_usage = None

        while tool_rounds < max_tool_rounds:
            tool_rounds += 1
            remaining_s = wall_clock_budget - (time.time() - start_time)
            if remaining_s < 15.0:
                logger.warning(
                    f"Wall-clock budget nearly exhausted ({remaining_s:.1f}s remaining). "
                    "Stopping tool loop to finalize migration."
                )
                break
            step_start = time.time()

            try:
                raw_msg, usage = call_llm(
                    client,
                    current_model,
                    messages=messages,
                    temperature=0.2,
                    tools=TOOL_DEFINITIONS,
                    timeout=min(remaining_s, settings.llm_timeout_s),
                )
            except AgentError as ae:
                logger.error(f"Agent LLM call failed: {ae}")
                steps.append(
                    AgentStepLog(
                        step=len(steps) + 1,
                        model=current_model,
                        action="call_llm_error",
                        details={"error": str(ae)},
                        latency_ms=(time.time() - step_start) * 1000,
                        status="error",
                    )
                )
                final_up = current_up_sql or (last_tool_sql if has_ddl_or_dml(last_tool_sql) else "")
                final_down = current_down_sql or (last_tool_down_sql if last_tool_down_sql else "")
                return AgentRunResult(
                    status="FAILED",
                    up_sql=final_up,
                    down_sql=final_down,
                    spec=spec,
                    steps=steps,
                    invariant_result=inv_result,
                    total_tokens=total_tokens,
                    total_latency_ms=(time.time() - start_time) * 1000,
                    stop_reason=f"AGENT_ERROR: {ae}",
                    notes=f"Agent LLM error: {ae}",
                    fast_model=fast_model,
                    ultra_model=ultra_model,
                    model_warnings=warnings,
                )
            last_step_lat = (time.time() - step_start) * 1000
            step_usage = usage

            if usage and hasattr(usage, "total_tokens"):
                total_tokens += usage.total_tokens

            content = raw_msg.content or ""
            tool_calls = getattr(raw_msg, "tool_calls", None)

            if content:
                extracted_up, extracted_down = parse_sql_scripts(content)
                if extracted_up:
                    step_up_sql = extracted_up
                    step_down_sql = extracted_down

            if tool_calls:
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
                    res = tool_executor.execute(fn_name, fn_args)

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
                            tokens_used=getattr(usage, "total_tokens", 0) if usage else 0,
                            latency_ms=last_step_lat,
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
                                    inv_result = InvariantResult.model_validate(res)
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

        # If tool rounds budget was exhausted after tool calls and no SQL produced yet,
        # prompt model once more so it can inspect tool outputs and deliver final SQL or submit (if budget permits).
        remaining_s = wall_clock_budget - (time.time() - start_time)
        if not step_up_sql and remaining_s >= 15.0 and messages and messages[-1].get("role") == "tool":
            step_start = time.time()
            final_reminder = {
                "role": "user",
                "content": (
                    "Tool round budget reached. Please now provide the final SQLite migration script "
                    "with both UP and DOWN migrations inside ```sql blocks, or call submit."
                ),
            }
            messages.append(final_reminder)
            try:
                raw_msg, usage = call_llm(
                    client,
                    current_model,
                    messages=messages,
                    temperature=0.2,
                    tools=TOOL_DEFINITIONS,
                    timeout=min(remaining_s, settings.llm_timeout_s),
                )
                last_step_lat = (time.time() - step_start) * 1000
                step_usage = usage
                if usage and hasattr(usage, "total_tokens"):
                    total_tokens += usage.total_tokens

                content = raw_msg.content or ""
                tool_calls = getattr(raw_msg, "tool_calls", None)

                if content:
                    extracted_up, extracted_down = parse_sql_scripts(content)
                    if extracted_up:
                        step_up_sql = extracted_up
                        step_down_sql = extracted_down

                if tool_calls:
                    for tc in tool_calls:
                        fn_obj = getattr(tc, "function", None)
                        fn_name = getattr(fn_obj, "name", "")
                        fn_args = getattr(fn_obj, "arguments", "")
                        if not isinstance(fn_args, str):
                            fn_args = json.dumps(fn_args)
                        res = tool_executor.execute(fn_name, fn_args)
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
                                tokens_used=getattr(usage, "total_tokens", 0) if usage else 0,
                                latency_ms=last_step_lat,
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
                            except Exception:
                                pass
            except AgentError as ae:
                logger.warning(f"Final tool completion call failed: {ae}")

        if not step_up_sql:
            last_raw_reply = content
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
                if last_tool_sql and has_ddl_or_dml(last_tool_sql):
                    step_up_sql = last_tool_sql
                    if not step_down_sql and last_tool_down_sql:
                        step_down_sql = last_tool_down_sql
                    consecutive_no_sql = 0
                else:
                    return AgentRunResult(
                        status="FAILED",
                        up_sql=current_up_sql,
                        down_sql=current_down_sql,
                        spec=spec,
                        steps=steps,
                        invariant_result=inv_result,
                        total_tokens=total_tokens,
                        total_latency_ms=(time.time() - start_time) * 1000,
                        stop_reason="MODEL_RETURNED_NO_SQL",
                        fast_model=fast_model,
                        ultra_model=ultra_model,
                        model_warnings=warnings,
                    )
            if not step_up_sql:
                logger.warning("Model produced empty SQL; retrying...")
                continue

        consecutive_no_sql = 0
        last_raw_reply = ""
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
                return AgentRunResult(
                    status="FAILED",
                    up_sql=current_up_sql,
                    down_sql=current_down_sql,
                    spec=spec,
                    steps=steps,
                    invariant_result=inv_result,
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

        # Dry-run on fresh clone
        notify("Executing dry-run verification on disposable sandbox clone...", {"step": step_num})
        base_clone = adapter.clone()
        test_clone = adapter.clone()
        try:
            run_ok, run_err, run_elapsed = adapter.run_in_sandbox(test_clone, current_up_sql, fk_enabled=True)

            if not run_ok:
                inv_result = InvariantResult(
                    status="FAILED",
                    passed=False,
                    error=run_err,
                    failed_checks=[run_err or "Execution error"],
                    elapsed_ms=run_elapsed,
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

        notify(
            f"Verification outcome: {inv_result.status} (Checks passed: {len(inv_result.passed_checks)}, failed: {len(inv_result.failed_checks)})",
            {"status": inv_result.status, "inv_result": inv_result.model_dump()},
        )

        if not inv_result.passed:
            seen_sql_attempts[sql_hash] = (step_num, list(inv_result.failed_checks))

        if inv_result.passed and inv_result.status in ("VERIFIED", "UNVERIFIED"):
            break

        # Escalate model to Ultra for next iteration
        current_model = ultra_model

    # Phase 3: Rollback Verification & Roundtrip Test
    rollback_ok = False
    rollback_err: str | None = None
    lossy_cols: list[str] = []
    if current_up_sql and current_down_sql:
        notify("Verifying rollback roundtrip on disposable clone...")
        base_clone = adapter.clone()
        try:
            rollback_ok, rollback_err, lossy_cols = verify_rollback_roundtrip(
                base_clone,
                current_up_sql,
                current_down_sql,
                target_table=target_table,
            )
        finally:
            base_clone.close()
        if not rollback_ok:
            notify(f"Rollback warning: {rollback_err}", {"lossy_columns": lossy_cols})
    elif current_up_sql:
        rollback_err = "no valid rollback script"

    # Estimate lock time
    lock_est = estimate_production_lock_time(
        inv_result.elapsed_ms if inv_result else 0.0,
        clone_rows=inv_result.before_count if inv_result else 0,
        production_rows=prod_rows,
    )

    final_status = "FAILED"
    if inv_result and inv_result.passed:
        final_status = inv_result.status  # VERIFIED or UNVERIFIED

    stop_reason = (
        "SUCCESS: All verification invariants passed."
        if final_status in ("VERIFIED", "UNVERIFIED")
        else "STEP_BUDGET_EXHAUSTED"
    )

    total_lat = (time.time() - start_time) * 1000

    return AgentRunResult(
        status=final_status,
        up_sql=current_up_sql,
        down_sql=current_down_sql,
        spec=spec,
        steps=steps,
        invariant_result=inv_result,
        rollback_verified=rollback_ok,
        rollback_error=rollback_err,
        lossy_columns=lossy_cols,
        lock_estimate=lock_est,
        total_tokens=total_tokens,
        total_latency_ms=total_lat,
        stop_reason=stop_reason,
        notes=notes,
        fast_model=fast_model,
        ultra_model=ultra_model,
        model_warnings=warnings,
    )

