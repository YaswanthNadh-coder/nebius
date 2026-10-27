"""Autonomous agent loop with tool-calling, tiered model escalation, and structured telemetry."""

import json
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from timemachine.adapters.base import DbAdapter
from timemachine.agent.llm import call_llm, clean_sql, get_client, resolve_nebius_models
from timemachine.agent.prompts import (
    build_migration_prompt,
    build_plan_prompt,
    build_repair_prompt,
)
from timemachine.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from timemachine.logging import get_logger
from timemachine.traps.generator import audit_traps
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
        step_start = time.time()
        plan_prompt = build_plan_prompt(schema, goal)
        raw_plan, _ = call_llm(client, fast_model, plan_prompt, temperature=0.1)
        plan_lat = (time.time() - step_start) * 1000

        # Parse JSON
        cleaned_json = raw_plan.strip()
        if "```json" in cleaned_json:
            cleaned_json = cleaned_json.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_json:
            cleaned_json = cleaned_json.split("```")[1].split("```")[0].strip()

        try:
            plan_dict = json.loads(cleaned_json)
            spec = MigrationSpec.model_validate(plan_dict)
            steps.append(
                AgentStepLog(
                    step=len(steps) + 1,
                    model=fast_model,
                    action="generate_spec",
                    details={"spec": spec.model_dump()},
                    latency_ms=plan_lat,
                    status="ok",
                )
            )
            notify("MigrationSpec formulated successfully.", {"spec": spec.model_dump()})
        except Exception as exc:
            logger.warning(f"Failed to parse model plan into MigrationSpec: {exc}. Using fallback spec.")
            spec = MigrationSpec(
                affected_tables=[target_table],
                row_policy="preserve_all",
                risky_operations=[],
            )

    tool_executor = ToolExecutor(adapter, spec=spec, target_table=target_table)

    # Phase 2: Agent Tool/Draft Loop
    current_model = fast_model
    current_up_sql = ""
    current_down_sql = ""
    notes = ""
    inv_result: InvariantResult | None = None
    step_num = 0

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
            )

        step_start = time.time()
        if step_num == 1:
            notify(f"Drafting initial migration using {current_model}...", {"step": step_num})
            prompt = build_migration_prompt(schema, spec, is_naive_baseline=is_naive_baseline)
        else:
            notify(f"Escalating repair to {current_model} with structured failure telemetry...", {"step": step_num})
            # Gather trap audit if available
            traps = inv_result.trap_audit if inv_result else []
            prompt = build_repair_prompt(schema, spec, current_up_sql, inv_result, trap_results=traps)

        content, tool_calls = call_llm(
            client,
            current_model,
            prompt,
            temperature=0.2,
            tools=TOOL_DEFINITIONS,
        )
        step_lat = (time.time() - step_start) * 1000

        step_up_sql = ""
        step_down_sql = ""

        # Case A: Model invoked tools
        if tool_calls:
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]
                res = tool_executor.execute(fn_name, fn_args)
                steps.append(
                    AgentStepLog(
                        step=len(steps) + 1,
                        model=current_model,
                        action=f"tool:{fn_name}",
                        details={"args": fn_args, "result": res},
                        latency_ms=step_lat,
                        status="ok" if "error" not in res else "error",
                    )
                )

                if fn_name == "submit":
                    step_up_sql = res.get("up_sql", "")
                    step_down_sql = res.get("down_sql", "")
                    notes = res.get("notes", "")
                elif fn_name == "run_on_clone":
                    inv_result = InvariantResult.model_validate(res)

        # Case B: Model returned direct text / code blocks
        if not step_up_sql and content:
            # Check for multiple blocks (e.g. up vs down)
            if "-- DOWN" in content or "ROLLBACK" in content:
                parts = content.split("-- DOWN")
                step_up_sql = clean_sql(parts[0])
                step_down_sql = clean_sql(parts[1]) if len(parts) > 1 else ""
            else:
                step_up_sql = clean_sql(content)

            steps.append(
                AgentStepLog(
                    step=len(steps) + 1,
                    model=current_model,
                    action="draft_migration" if step_num == 1 else "repair_migration",
                    details={"up_sql_preview": step_up_sql[:200]},
                    latency_ms=step_lat,
                    status="ok",
                )
            )

        if not step_up_sql:
            logger.warning("Model produced empty SQL; retrying...")
            continue

        current_up_sql = step_up_sql
        if step_down_sql:
            current_down_sql = step_down_sql

        # Dry-run on fresh clone
        notify("Executing dry-run verification on disposable sandbox clone...", {"step": step_num})
        base_clone = adapter.clone()
        test_clone = adapter.clone()
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
            # Evaluate full invariants
            inv_result = evaluate_migration_invariants(
                base_clone,
                test_clone,
                target_table=target_table,
                spec=spec,
                elapsed_ms=run_elapsed,
            )
            # Evaluate adversarial traps if injected
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

        base_clone.close()
        test_clone.close()

        notify(
            f"Verification outcome: {inv_result.status} (Checks passed: {len(inv_result.passed_checks)}, failed: {len(inv_result.failed_checks)})",
            {"status": inv_result.status, "inv_result": inv_result.model_dump()},
        )

        if inv_result.passed and inv_result.status in ("VERIFIED", "UNVERIFIED"):
            # Migration succeeded!
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
        rollback_ok, rollback_err, lossy_cols = verify_rollback_roundtrip(
            base_clone,
            current_up_sql,
            current_down_sql,
            target_table=target_table,
        )
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

    stop_reason = "SUCCESS: All verification invariants passed." if final_status in ("VERIFIED", "UNVERIFIED") else "STEP_BUDGET_EXHAUSTED"

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
    )
