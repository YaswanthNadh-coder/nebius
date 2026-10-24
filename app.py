import os
import streamlit as st
from dotenv import load_dotenv
from scenarios import SCENARIOS
from agent import run_time_machine_agent

load_dotenv()

st.set_page_config(page_title="SQL Migration Time Machine", page_icon="⏳", layout="wide")

st.title("⏳ SQL Migration 'Dry-Run Time Machine'")
st.caption(
    "Nebius x NVIDIA Global AI Hackathon (Coding & Agentic Engineering Track) • "
    "Tiered NVIDIA Nemotron Routing on Nebius Token Factory + In-Memory SQLite Sandboxing"
)

with st.sidebar:
    st.header("⚙️ Agent & Sandbox Settings")
    env_key = os.environ.get("NEBIUS_API_KEY", "")
    api_key_input = st.text_input(
        "Nebius API Key (loaded from .env if set)",
        value=env_key,
        type="password",
        help="Your Nebius Token Factory API key.",
    )
    num_rows = st.slider("Synthetic Production Rows", min_value=500, max_value=5000, value=1000, step=500)
    max_retries = st.slider("Max Self-Healing Attempts", min_value=2, max_value=5, value=3)
    st.divider()
    st.markdown(
        "**How Tiered Routing Works:**\n"
        "- **Attempt #1:** `Nemotron 3 Super` drafts fast initial migration SQL.\n"
        "- **RAM Clone Audit:** `sqlite3.backup()` clones DB in <5ms and verifies row counts & NULL constraints.\n"
        "- **Self-Heal Loop:** `Nemotron 3 Ultra` analyzes exact row-loss diffs and rewrites lossless SQL."
    )

scenario_name = st.selectbox("Select Pre-Built Adversarial Scenario", list(SCENARIOS.keys()))
active_scenario = SCENARIOS[scenario_name]

col1, col2 = st.columns(2)
with col1:
    schema_input = st.text_area("1. Production SQL Schema", value=active_scenario["schema"], height=210)
with col2:
    goal_input = st.text_area("2. Desired Schema Migration Goal", value=active_scenario["goal"], height=210)

if st.button("🚀 Run Adversarial Time-Machine Dry Run", type="primary", use_container_width=True):
    if not api_key_input or api_key_input == "your_nebius_api_key_here":
        st.error("Please add your real NEBIUS_API_KEY in `.env` or paste it in the left sidebar first!")
    else:
        with st.spinner(f"Seeding {num_rows:,} dirty rows in RAM & executing Write -> Test -> Self-Heal loop..."):
            history, trap_stats = run_time_machine_agent(
                schema_input,
                goal_input,
                num_rows=num_rows,
                max_retries=max_retries,
                api_key_override=api_key_input,
            )

        st.subheader("🧪 Injected Adversarial Production Traps")
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Total Seeded Rows", f"{trap_stats['total_rows']:,}")
        t2.metric("NULL Dept Codes", trap_stats["null_dept_rows"])
        t3.metric("Unmapped Legacy Depts", trap_stats["unmapped_legacy_dept_rows"])
        t4.metric("Single-Word Names", trap_stats["single_word_name_rows"])
        t5.metric("NULL Salary Strings", trap_stats["null_salary_rows"])

        st.subheader("🔄 Agent Execution & Self-Healing Timeline")
        for step in history:
            badge = (
                f"✅ PASSED ({step['after_count']:,} / {step['before_count']:,} Rows Verified in {step['elapsed_ms']}ms)"
                if step["passed"]
                else f"❌ FAILED SANDBOX DRY-RUN ({step['elapsed_ms']}ms)"
            )
            with st.expander(f"Attempt #{step['attempt']} — Model: `{step['model']}` — {badge}", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows Before Migration", f"{step['before_count']:,}")
                c2.metric(
                    "Rows After Migration",
                    f"{step['after_count']:,}",
                    delta=step["after_count"] - step["before_count"],
                )
                c3.metric("Active Nemotron Model", step["model"].split("/")[-1])

                st.code(step["sql"], language="sql")

                if step["error"]:
                    st.error(f"**Sandbox Audit Caught Data Corruption:** {step['error']}")
                    st.info("⚡ Escalating to **Nemotron 3 Ultra** with failure telemetry to synthesize a lossless patch...")
                else:
                    st.success(
                        f"🎉 **Mathematical Verification Complete:** All {step['after_count']:,} dirty rows migrated with 0 rows lost and 0 NULL corruptions!"
                    )
                    st.write("**Sample Verified Output Rows (from in-memory clone):**")
                    st.dataframe(step["sample_rows"], use_container_width=True)
