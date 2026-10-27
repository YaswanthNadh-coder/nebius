"""Live smoke test script validating dry-run time machine against real Nebius endpoint.

Executes spec generation and migration dry-runs for built-in scenarios 1 and 2,
testing both standard runs and naive baselines.
Outputs structured summary to live_smoke_out.json. Never runs in CI.
"""

import json
import os
import sys
import time
from typing import Any

# Reconfigure stdout/stderr for utf-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from timemachine.agent.llm import get_client, resolve_nebius_models
from timemachine.agent.loop import AgentRunResult, run_time_machine_agent
from timemachine.agent.planner import generate_spec, validate_spec_against_schema
from timemachine.config import get_api_key_info, is_placeholder_key, load_env_file
from timemachine.traps.profiler import profile_database
from timemachine.ui.app_view import create_builtin_scenario_adapter


def inspect_model_behavior(result: AgentRunResult) -> dict[str, Any]:
    """Inspect model behavior from run steps and raw responses."""
    had_tool_calls = False
    emitted_think = False
    followed_down_convention = False

    for s in result.steps:
        if s.action.startswith("tool:"):
            had_tool_calls = True
        raw_rep = s.details.get("raw_reply", "")
        if "<think>" in raw_rep.lower() or "</think>" in raw_rep.lower():
            emitted_think = True

    if "-- DOWN" in result.up_sql or "-- DOWN" in result.down_sql or result.down_sql:
        followed_down_convention = True

    return {
        "had_tool_calls": had_tool_calls,
        "emitted_think": emitted_think,
        "followed_down_convention": followed_down_convention,
    }


def main():
    print("=" * 70)
    print("[START] SQL MIGRATION TIME MACHINE - LIVE NEBIUS API SMOKE TEST")
    print("=" * 70)

    load_env_file()
    key, src, masked = get_api_key_info()

    if is_placeholder_key(key):
        print("[ERROR] NEBIUS_API_KEY is not configured or is a placeholder.")
        print(f"Detected key status: {masked} from {src}")
        sys.exit(1)

    print(f"[KEY] Nebius Key resolved from: {src} ({masked})")

    try:
        client = get_client(key)
        fast_model, ultra_model, warnings = resolve_nebius_models(client)
        print(f"[MODEL] Fast Drafting Model: {fast_model}")
        print(f"[MODEL] Deep Repair Model:   {ultra_model}")
        if warnings:
            for w in warnings:
                print(f"[WARN] Warning: {w}")
    except Exception as exc:
        print(f"[ERROR] Failed to connect to Nebius API / resolve models: {exc}")
        sys.exit(1)

    scenarios = [
        {
            "id": 1,
            "name": "Scenario 1: Dept Normalization & Split",
            "goal": "Normalize department into a separate departments table and split full_name into first_name and last_name.",
            "target_table": "employees",
        },
        {
            "id": 2,
            "name": "Scenario 2: Legacy Salary Banding",
            "goal": "Clean raw_salary by converting string amounts ($120,000) to integer cents in salary_cents, and compute salary_band.",
            "target_table": "employees",
        },
    ]

    all_results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fast_model": fast_model,
        "ultra_model": ultra_model,
        "warnings": warnings,
        "scenarios": {},
    }

    try:
        for sc in scenarios:
            sc_id = sc["id"]
            sc_name = sc["name"]
            goal = sc["goal"]
            target_t = sc["target_table"]

            print(f"\n{'='*70}\n📋 Running {sc_name}\n{'='*70}")
            sc_record: dict[str, Any] = {"name": sc_name, "goal": goal}

            # 1. Generate Spec
            adapter = create_builtin_scenario_adapter(sc_id)
            schema = adapter.introspect()
            clone = adapter.clone()
            try:
                profiles = profile_database(clone, schema)
            except Exception:
                profiles = {}
            finally:
                clone.close()

            print("\n--- Phase 1: generate_spec ---")
            spec_res = generate_spec(
                client=client,
                model=fast_model,
                schema=schema,
                goal=goal,
                profiles=profiles,
                target_table=target_t,
            )

            val_errs: list[str] = []
            num_mappings = 0
            if spec_res.spec:
                val_errs = validate_spec_against_schema(spec_res.spec, schema, target_t)
                tbl_mappings = spec_res.spec.get_table_mappings(target_t)
                num_mappings = len(tbl_mappings)

            print(f"  Source:             {spec_res.source}")
            print(f"  Model:              {spec_res.model}")
            print(f"  Latency:            {spec_res.latency_ms:.1f}ms")
            print(f"  Column Mappings:    {num_mappings}")
            print(f"  Validation Errors:  {val_errs if val_errs else 'None'}")
            if spec_res.error:
                print(f"  Error:              {spec_res.error}")

            sc_record["spec"] = {
                "source": spec_res.source,
                "model": spec_res.model,
                "latency_ms": spec_res.latency_ms,
                "num_mappings": num_mappings,
                "validation_errors": val_errs,
                "error": spec_res.error,
                "raw_response": spec_res.raw_response[:500] if spec_res.raw_response else "",
            }

            # 2. Standard Agent Run
            print("\n--- Phase 2: Standard Agent Run (is_naive_baseline=False) ---")
            std_adapter = create_builtin_scenario_adapter(sc_id)
            std_res = run_time_machine_agent(
                adapter=std_adapter,
                goal=goal,
                spec=spec_res.spec,
                target_table=target_t,
                api_key_override=key,
                is_naive_baseline=False,
            )
            std_behavior = inspect_model_behavior(std_res)

            print(f"  Status:             {std_res.status}")
            print(f"  Stop Reason:        {std_res.stop_reason}")
            print(f"  Step Actions:       {[s.action for s in std_res.steps]}")
            print(f"  Total Tokens:       {std_res.total_tokens}")
            fc = std_res.invariant_result.failed_checks if std_res.invariant_result else []
            print(f"  Failed Checks:      {fc}")
            print(f"  Rollback Verified:  {std_res.rollback_verified}")
            print(f"  Rollback Error:     {std_res.rollback_error}")
            print(f"  Tool Calls Used:    {std_behavior['had_tool_calls']}")
            print(f"  Emitted <think>:    {std_behavior['emitted_think']}")
            print(f"  Down Convention:    {std_behavior['followed_down_convention']}")
            print(f"  UP SQL Preview:     \n{std_res.up_sql[:400]}")
            print(f"  DOWN SQL Preview:   \n{std_res.down_sql[:400]}")

            sc_record["standard_run"] = {
                "status": std_res.status,
                "stop_reason": std_res.stop_reason,
                "actions": [s.action for s in std_res.steps],
                "total_tokens": std_res.total_tokens,
                "failed_checks": fc,
                "rollback_verified": std_res.rollback_verified,
                "rollback_error": std_res.rollback_error,
                "behavior": std_behavior,
                "up_sql_preview": std_res.up_sql[:400],
                "down_sql_preview": std_res.down_sql[:400],
            }

            # 3. Naive Baseline Run
            print("\n--- Phase 3: Naive Baseline Run (is_naive_baseline=True) ---")
            naive_adapter = create_builtin_scenario_adapter(sc_id)
            naive_res = run_time_machine_agent(
                adapter=naive_adapter,
                goal=goal,
                spec=spec_res.spec,
                target_table=target_t,
                api_key_override=key,
                is_naive_baseline=True,
            )
            naive_behavior = inspect_model_behavior(naive_res)

            print(f"  Status:             {naive_res.status}")
            print(f"  Stop Reason:        {naive_res.stop_reason}")
            print(f"  Step Actions:       {[s.action for s in naive_res.steps]}")
            print(f"  Total Tokens:       {naive_res.total_tokens}")
            n_fc = naive_res.invariant_result.failed_checks if naive_res.invariant_result else []
            print(f"  Failed Checks:      {n_fc}")
            print(f"  Rollback Verified:  {naive_res.rollback_verified}")
            print(f"  Rollback Error:     {naive_res.rollback_error}")
            print(f"  Tool Calls Used:    {naive_behavior['had_tool_calls']}")
            print(f"  Emitted <think>:    {naive_behavior['emitted_think']}")
            print(f"  Down Convention:    {naive_behavior['followed_down_convention']}")
            print(f"  UP SQL Preview:     \n{naive_res.up_sql[:400]}")
            print(f"  DOWN SQL Preview:   \n{naive_res.down_sql[:400]}")

            sc_record["naive_baseline_run"] = {
                "status": naive_res.status,
                "stop_reason": naive_res.stop_reason,
                "actions": [s.action for s in naive_res.steps],
                "total_tokens": naive_res.total_tokens,
                "failed_checks": n_fc,
                "rollback_verified": naive_res.rollback_verified,
                "rollback_error": naive_res.rollback_error,
                "behavior": naive_behavior,
                "up_sql_preview": naive_res.up_sql[:400],
                "down_sql_preview": naive_res.down_sql[:400],
            }

            all_results["scenarios"][f"scenario_{sc_id}"] = sc_record
            out_path = os.path.join(os.path.dirname(__file__), "..", "live_smoke_out.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

        print(f"\n[SUCCESS] Live smoke test completed successfully. Full log saved to: {out_path}")

    except Exception as exc:
        print(f"\n[FAILED] Live smoke test failed with exception: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        out_path = os.path.join(os.path.dirname(__file__), "..", "live_smoke_out.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
        except Exception:
            pass


if __name__ == "__main__":
    main()
