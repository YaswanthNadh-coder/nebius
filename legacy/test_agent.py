"""CLI smoke test to verify Nebius API connection and the SQLite Time Machine loop."""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from legacy.agent import get_client, resolve_nebius_models, run_time_machine_agent
    from legacy.scenarios import SCENARIOS
except ImportError:
    from agent import get_client, resolve_nebius_models, run_time_machine_agent
    from scenarios import SCENARIOS
from timemachine.config import get_api_key_info, is_placeholder_key, load_env_file

load_env_file()

if __name__ == "__main__":
    clean_key, key_src, _ = get_api_key_info()
    if is_placeholder_key(clean_key):
        print(f"[ERROR] No valid NEBIUS_API_KEY found (checked {key_src}). Set key in .env before running live tests.", file=sys.stderr)
        sys.exit(1)

    try:
        client = get_client()
        fast_m, ultra_m = resolve_nebius_models(client)
        print(f"✅ Connected to Nebius Token Factory! Fast Model: {fast_m} | Ultra Model: {ultra_m}")

        sc = SCENARIOS["Scenario 1: Employee Table Rebuild (NULL & Unmatched Foreign Key Trap)"]
        history, stats = run_time_machine_agent(sc["schema"], sc["goal"], num_rows=200, max_retries=3)

        passed = False
        for h in history:
            print(f"Attempt #{h['attempt']} ({h['model']}): Passed={h['passed']} | Rows={h['after_count']}/{h['before_count']}")
            if h["passed"]:
                passed = True

        if passed:
            print("🎉 Smoke test passed!")
            sys.exit(0)
        else:
            print("[FAIL] Migration failed all verification attempts.", file=sys.stderr)
            sys.exit(1)

    except Exception as exc:
        print(f"[ERROR] Smoke test encountered exception: {exc}", file=sys.stderr)
        sys.exit(1)
