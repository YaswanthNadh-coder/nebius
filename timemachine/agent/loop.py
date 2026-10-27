"""Autonomous agent loop with tool-calling, tiered model escalation, and structured telemetry."""

import json
import re
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from timemachine.adapters.base import DbAdapter
from timemachine.agent.llm import call_llm, clean_sql, get_client, resolve_nebius_models
from timemachine.agent.planner import generate_spec
from timemachine.agent.prompts import (
    build_migration_prompt,
    build_repair_prompt,
)
from timemachine.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from timemachine.config import get_settings
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


def parse_sql_scripts(content: str) -> tuple[str, str]:
    """Parse UP and optional DOWN migration scripts from LLM content."""
    if "-- DOWN" in content:
        parts = content.split("-- DOWN", 1)
        up = clean_sql(parts[0])
        down = clean_sql(parts[1])
        return up, down

    fence_re = re.compile(
        r"^```(?:sql|sqlite)?\s*$\n(.*?)^```\s*$",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    blocks = fence_re.findall(content.replace("\r\n", "\n"))
    valid_blocks = [b.strip() for b in blocks if b.strip()]
    if len(valid_blocks) >= 2:
        return clean_sql(valid_blocks[0]), clean_sql(valid_blocks[1])

    return clean_sql(content), ""


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

    # Phase 1: Planning (derive spec if not provided)
    if spec is None:
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
    current_model = fast_model
    current_up_sql = ""
    current_down_sql = ""
    notes = ""
    inv_result: InvariantResult | None = None
    step_num = 0
    consecutive_no_sql = 0
    last_raw_reply = ""

    while step_num < max_steps:
        step_num += 1
        elapsed_total = time.time() - start_time
        if elapsed_total > 180.0:
            return AgentRunResult(
                status="FAILED",
                up_sql=current_up_sql,
                down_sql=current_down_sql,
                spec=spec,
                steps=steps,
                invariant_result=inv_result,
                total_tokens=total_tokens,
                total_latency_ms=elapsed_total * 1000,
                stop_reason="WALL_CLOCK_TIMEOUT: Exceeded 180s budget.",
                fast_model=fast_model,
                ultra_model=ultra_model,
                model_warnings=warnings,
            )

        if inv_result is None:
            if consecutive_no_sql > 0 and last_raw_reply:
                notify(f"Re-prompting {current_model} for migration SQL...", {"step": step_num})
                prompt = build_repair_prompt(schema, spec, last_raw_reply, None)
            else:
                notify(f"Drafting initial migration using {current_model}...", {"step": step_num})
                prompt = build_migration_prompt(schema, spec, is_naive_baseline=is_naive_baseline)
        else:
            notify(f"Escalating repair to {current_model} with structured failure telemetry...", {"step": step_num})
            traps = inv_result.trap_audit if inv_result else []
            prompt = build_repair_prompt(schema, spec, current_up_sql, inv_result, trap_results=traps)

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        step_up_sql = ""
        step_down_sql = ""
        tool_rounds = 0
        content = ""
        last_step_lat = 0.0
        step_usage = None

        while tool_rounds < 4:
            tool_rounds += 1
            remaining_s = max(1.0, 180.0 - (time.time() - start_time))
            step_start = time.time()

            raw_msg, usage = call_llm(
                client,
                current_model,
                messages=messages,
                temperature=0.2,
                tools=TOOL_DEFINITIONS,
                timeout=min(remaining_s, get_settings().llm_timeout_s),
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
                        inv_result = InvariantResult.model_validate(res)

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
                break

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
            logger.warning("Model produced empty SQL; retrying...")
            continue

        consecutive_no_sql = 0
        last_raw_reply = ""
        current_up_sql = step_up_sql
        if step_down_sql:
            current_down_sql = step_down_sql

        steps.append(
            AgentStepLog(
                step=len(steps) + 1,
                model=current_model,
                action="draft_migration" if inv_result is None else "repair_migration",
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

