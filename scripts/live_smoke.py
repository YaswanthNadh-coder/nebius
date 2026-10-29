"""Live smoke test script validating dry-run time machine against real Nebius endpoint.

Executes spec generation and migration dry-runs for built-in scenarios 1 and 2,
testing both standard runs and naive baselines.
Outputs structured summary to live_smoke_out.json. Never runs in CI.
"""

import json
import os
import sqlite3
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

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.agent.llm import get_client, resolve_nebius_models
from timemachine.agent.loop import AgentRunResult, run_time_machine_agent
from timemachine.agent.planner import generate_spec, validate_spec_against_schema
from timemachine.config import get_api_key_info, get_settings, is_placeholder_key, load_env_file
from timemachine.traps.profiler import profile_database, profile_table
from timemachine.ui.app_view import create_builtin_scenario_adapter


def print_attempts_table(attempts: list[Any]) -> None:
    """Print nicely formatted table of migration verification attempts."""
    if not attempts:
        print("  Attempts: None")
        return
    print("\n  +---------+----------------------+--------+--------+------------------------------------+--------------------+------------+--------+")
    print("  | Attempt | Model                | Passed | Failed | First Failed Check                 | Sandbox Error      | Latency ms | Tokens |")
    print("  +---------+----------------------+--------+--------+------------------------------------+--------------------+------------+--------+")
    for a in attempts:
        m_short = getattr(a, "model", "").split("/")[-1][:20]
        ff = (getattr(a, "first_failed_check", "") or "None")[:34]
        sb = (getattr(a, "sandbox_error", None) or "None")[:18]
        att = getattr(a, "attempt", 0)
        p = getattr(a, "passed_count", 0)
        f = getattr(a, "failed_count", 0)
        lat = getattr(a, "latency_ms", 0.0)
        tok = getattr(a, "tokens", 0)
        print(f"  | {att:^7} | {m_short:<20} | {p:^6} | {f:^6} | {ff:<34} | {sb:<18} | {lat:^10.1f} | {tok:^6} |")
    print("  +---------+----------------------+--------+--------+------------------------------------+--------------------+------------+--------+\n")


def serialize_attempts(attempts: list[Any]) -> list[dict[str, Any]]:
    """Serialize AttemptRecord list for live_smoke_out.json."""
    return [
        {
            "attempt": getattr(a, "attempt", 0),
            "model": getattr(a, "model", ""),
            "up_sql_preview": (getattr(a, "up_sql", "") or "")[:400],
            "down_sql_preview": (getattr(a, "down_sql", "") or "")[:400],
            "split_rule": getattr(a, "split_rule", "") or "unknown",
            "passed_count": getattr(a, "passed_count", 0),
            "failed_count": getattr(a, "failed_count", 0),
            "first_failed_check": getattr(a, "first_failed_check", ""),
            "failed_checks": [getattr(a, "first_failed_check", "")] if getattr(a, "first_failed_check", "") else [],
            "sandbox_error": getattr(a, "sandbox_error", None),
            "latency_ms": getattr(a, "latency_ms", 0.0),
            "tokens": getattr(a, "tokens", 0),
            "tool_rounds_used": getattr(a, "tool_rounds_used", 0),
            "no_sql_replies": getattr(a, "no_sql_replies", 0),
            "raw_reply_preview": (getattr(a, "raw_reply", "") or "")[:500],
        }
        for a in attempts
    ]


def assert_no_up_contains_down(attempts: list[Any], run_label: str = "run") -> None:
    """Fail live smoke execution if any attempt's UP migration contains the DOWN script."""
    import re
    for a in attempts:
        up = getattr(a, "up_sql", "") or ""
        down = getattr(a, "down_sql", "") or ""
        att = getattr(a, "attempt", 0)
        if not up or not down:
            continue
        cleaned_down = re.sub(r"--.*$", "", down, flags=re.MULTILINE).strip()
        cleaned_up = re.sub(r"--.*$", "", up, flags=re.MULTILINE).strip()
        if len(cleaned_down) > 40 and cleaned_down in cleaned_up:
            raise AssertionError(
                f"[{run_label}] Attempt #{att}: up_sql contains down_sql! Splitter failed to separate UP and DOWN."
            )
        if re.search(r"--\s*[-=*~_# ]*(?:down|rollback|revert|undo|reverse|backward|migrate\s+down)\b", up, re.IGNORECASE):
            raise AssertionError(
                f"[{run_label}] Attempt #{att}: up_sql contains an un-split DOWN marker line!"
            )


def benchmark_performance(row_counts: list[int] | None = None) -> dict[str, Any]:
    """Measure and print clone / profile / run_on_clone timings at synthetic row counts (10k and 200k rows)."""
    if row_counts is None:
        row_counts = [10_000, 200_000]

    print("\n" + "=" * 70)
    print("⚡ PERFORMANCE BENCHMARK (clone / profile / run_on_clone)")
    print("=" * 70)

    results: dict[str, Any] = {}

    for count in row_counts:
        print(f"\n[BENCH] Seeding database with {count:,} rows...")
        t0_seed = time.time()
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        cur.execute("PRAGMA synchronous = OFF;")
        cur.execute("PRAGMA journal_mode = MEMORY;")
        cur.execute("""
            CREATE TABLE bench (
                id INTEGER PRIMARY KEY,
                name TEXT,
                salary INTEGER,
                created_at TEXT
            );
        """)
        batch_size = 10_000
        for b in range(0, count, batch_size):
            cur.executemany(
                "INSERT INTO bench (id, name, salary, created_at) VALUES (?, ?, ?, ?)",
                [(i, f"Person_{i}", 50000 + (i % 50000), "2026-01-01") for i in range(b + 1, min(b + batch_size + 1, count + 1))]
            )
        conn.commit()
        seed_ms = (time.time() - t0_seed) * 1000

        adapter = SQLiteAdapter.create_in_memory()
        adapter_conn = adapter.open_readonly()
        conn.backup(adapter_conn)

        # 1. Clone timing
        t0_clone = time.time()
        clone = adapter.clone()
        clone_ms = (time.time() - t0_clone) * 1000

        # 2. Profile table timing (sampling capped at 50k rows)
        t0_prof = time.time()
        prof = profile_table(clone, "bench")
        prof_ms = (time.time() - t0_prof) * 1000

        # 3. run_on_clone timing (sandbox execution)
        t0_run = time.time()
        ok, err, dur = adapter.run_in_sandbox(clone, "UPDATE bench SET salary = salary + 10 WHERE id <= 1000;", fk_enabled=True)
        run_ms = (time.time() - t0_run) * 1000

        clone.close()
        conn.close()

        sampled = prof.get("sampled", False) if isinstance(prof, dict) else False
        results[f"{count}_rows"] = {
            "rows": count,
            "seed_ms": round(seed_ms, 2),
            "clone_ms": round(clone_ms, 2),
            "profile_ms": round(prof_ms, 2),
            "run_on_clone_ms": round(run_ms, 2),
            "sampled": sampled,
        }

    print("\n  +-----------+------------+---------------+------------------+------------+")
    print("  | Rows      | Clone (ms) | Profile (ms)  | RunOnClone (ms)  | Sampled    |")
    print("  +-----------+------------+---------------+------------------+------------+")
    for k, v in results.items():
        print(f"  | {v['rows']:<9,}| {v['clone_ms']:<10.2f} | {v['profile_ms']:<13.2f} | {v['run_on_clone_ms']:<16.2f} | {str(v['sampled']):<10} |")
    print("  +-----------+------------+---------------+------------------+------------+\n")

    return results


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

    # Note: API key is never printed, only masked status is shown
    print(f"[KEY] Nebius Key resolved from: {src} ({masked})")

    settings = get_settings(reload=True)
    print(f"[CONFIG] Max Tokens Per Call: {settings.max_tokens_per_call}")
    print(f"[CONFIG] Reasoning Effort:    {settings.reasoning_effort or 'default'}")

    # Phase 0: Performance Benchmarks (at 10k and 200k rows)
    bench_results = benchmark_performance()

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
        "performance_benchmarks": bench_results,
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
            assert_no_up_contains_down(std_res.attempts, f"{sc_name} standard")

            print(f"  Status:                  {std_res.status}")
            print(f"  Stop Reason:             {std_res.stop_reason}")
            print(f"  Step Actions:            {[s.action for s in std_res.steps]}")
            print(f"  Total Tokens:            {std_res.total_tokens}")
            fc = std_res.invariant_result.failed_checks if std_res.invariant_result else []
            print(f"  Failed Checks:           {fc}")
            print(f"  Rollback Classification: {std_res.rollback_classification or 'NONE'}")
            print(f"  Rollback Verified:       {std_res.rollback_verified}")
            print(f"  Rollback Error:          {std_res.rollback_error}")
            print(f"  Tool Calls Used:         {std_behavior['had_tool_calls']}")
            print(f"  Emitted <think>:         {std_behavior['emitted_think']}")
            print(f"  Down Convention:         {std_behavior['followed_down_convention']}")
            print_attempts_table(std_res.attempts)
            print(f"  UP SQL Preview:     \n{std_res.up_sql[:400]}")
            print(f"  DOWN SQL Preview:   \n{std_res.down_sql[:400]}")

            sc_record["standard_run"] = {
                "status": std_res.status,
                "stop_reason": std_res.stop_reason,
                "actions": [s.action for s in std_res.steps],
                "attempts": serialize_attempts(std_res.attempts),
                "total_tokens": std_res.total_tokens,
                "failed_checks": fc,
                "rollback_classification": std_res.rollback_classification,
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
            assert_no_up_contains_down(naive_res.attempts, f"{sc_name} naive")

            print(f"  Status:                  {naive_res.status}")
            print(f"  Stop Reason:             {naive_res.stop_reason}")
            print(f"  Step Actions:            {[s.action for s in naive_res.steps]}")
            print(f"  Total Tokens:            {naive_res.total_tokens}")
            n_fc = naive_res.invariant_result.failed_checks if naive_res.invariant_result else []
            print(f"  Failed Checks:           {n_fc}")
            print(f"  Rollback Classification: {naive_res.rollback_classification or 'NONE'}")
            print(f"  Rollback Verified:       {naive_res.rollback_verified}")
            print(f"  Rollback Error:          {naive_res.rollback_error}")
            print(f"  Tool Calls Used:         {naive_behavior['had_tool_calls']}")
            print(f"  Emitted <think>:         {naive_behavior['emitted_think']}")
            print(f"  Down Convention:         {naive_behavior['followed_down_convention']}")
            print_attempts_table(naive_res.attempts)
            print(f"  UP SQL Preview:     \n{naive_res.up_sql[:400]}")
            print(f"  DOWN SQL Preview:   \n{naive_res.down_sql[:400]}")

            sc_record["naive_baseline_run"] = {
                "status": naive_res.status,
                "stop_reason": naive_res.stop_reason,
                "actions": [s.action for s in naive_res.steps],
                "attempts": serialize_attempts(naive_res.attempts),
                "total_tokens": naive_res.total_tokens,
                "failed_checks": n_fc,
                "rollback_classification": naive_res.rollback_classification,
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
