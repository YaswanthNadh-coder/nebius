"""CLI smoke test to verify Nebius API connection and the SQLite Time Machine loop."""
import os
import sys
from dotenv import load_dotenv
from scenarios import SCENARIOS
from agent import get_client, resolve_nebius_models, run_time_machine_agent

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

if __name__ == "__main__":
    key = os.environ.get("NEBIUS_API_KEY", "")
    if not key or key == "your_nebius_api_key_here":
        print("⚠️  Set your real NEBIUS_API_KEY in .env before running live LLM calls.")
    else:
        client = get_client()
        fast_m, ultra_m = resolve_nebius_models(client)
        print(f"✅ Connected to Nebius Token Factory! Fast Model: {fast_m} | Ultra Model: {ultra_m}")
        sc = SCENARIOS["Scenario 1: Employee Table Rebuild (NULL & Unmatched Foreign Key Trap)"]
        history, stats = run_time_machine_agent(sc["schema"], sc["goal"], num_rows=500, max_retries=3)
        for h in history:
            print(f"Attempt #{h['attempt']} ({h['model']}): Passed={h['passed']} | Rows={h['after_count']}/{h['before_count']}")
