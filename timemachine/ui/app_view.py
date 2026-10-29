"""Streamlit UI view rendering the 5-tab autonomous migration time machine."""

import hashlib
import json
import os
from typing import Any

import streamlit as st

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.agent.llm import get_client, resolve_nebius_models
from timemachine.agent.loop import AgentRunResult, run_time_machine_agent
from timemachine.agent.planner import generate_spec
from timemachine.config import (
    AgentError,
    get_api_key_info,
    is_placeholder_key,
    load_env_file,
)
from timemachine.logging import get_logger
from timemachine.report.history import load_run_history, save_run_history
from timemachine.report.writer import (
    export_alembic,
    generate_markdown_report,
    wrap_transaction_with_prechecks,
)
from timemachine.scenarios import get_builtin_scenario_reference, validate_spec_satisfiability
from timemachine.traps.profiler import (
    get_cached_profile,
    mask_value,
    profile_database,
    set_cached_profile,
)
from timemachine.verify.spec import MigrationSpec

logger = get_logger("ui.app_view")


@st.cache_data
def get_cached_introspect(fingerprint: str, _adapter: Any) -> Any:
    """Cache introspected database schema keyed by database source fingerprint."""
    return _adapter.introspect()


def init_session_state():
    """Ensure persistent session state variables exist."""
    defaults = {
        "adapter": None,
        "adapter_name": "Built-in: Employees Normalization",
        "target_table": "employees",
        "goal": "Split full_name into first_name/last_name and add a foreign key department_code -> departments(dept_code); orphan codes and NULLs become NULL.",
        "spec": None,
        "spec_approved": False,
        "spec_source": "",
        "spec_model": "",
        "spec_latency_ms": 0.0,
        "spec_goal_hash": "",
        "run_result": None,
        "is_naive_baseline": False,
        "fk_enabled": True,
        "inject_traps": True,
        "api_key_override": "",
        "active_tab": 0,
        "source_fingerprint": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def set_source(adapter: SQLiteAdapter, name: str, fingerprint: Any):
    """Set active database source adapter and fingerprint; reset dependent plan state."""
    st.session_state["adapter"] = adapter
    st.session_state["adapter_name"] = name
    st.session_state["source_fingerprint"] = fingerprint
    st.session_state["spec"] = None
    st.session_state["spec_approved"] = False
    st.session_state["run_result"] = None


def create_builtin_scenario_adapter(scenario: int = 1) -> SQLiteAdapter:
    """Create in-memory SQLiteAdapter pre-seeded for demo scenarios."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    cur = conn.cursor()

    if scenario == 1:
        # Scenario 1: Department Normalization & Name Split
        cur.execute("""
            CREATE TABLE departments (
                dept_code TEXT PRIMARY KEY,
                dept_name TEXT NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                department TEXT,
                salary INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.executemany(
            "INSERT INTO departments (dept_code, dept_name) VALUES (?, ?);",
            [("ENG", "Engineering"), ("MKT", "Marketing"), ("HR", "Human Resources"), ("FIN", "Finance")],
        )
        cur.executemany(
            "INSERT INTO employees (full_name, department, salary) VALUES (?, ?, ?);",
            [
                ("Alice Smith", "ENG", 95000),
                ("Bob Jones", "MKT", 72000),
                ("Charlie Brown", "HR", 68000),
                ("Diana Prince", "LEGACY_99", 110000),  # Orphan foreign key edge case
                ("Eve", "ENG", 89000),                  # Mononym edge case
                ("Frank O'Connor", None, 75000),        # Null department edge case
                ("Grace Hopper 🚀", "ENG", 125000),     # Unicode & emoji
            ],
        )
        # Create an index and trigger to test preservation
        cur.execute("CREATE INDEX idx_emp_salary ON employees(salary);")
        cur.execute("""
            CREATE TRIGGER trg_emp_audit AFTER INSERT ON employees
            BEGIN
                SELECT 1;
            END;
        """)
        conn.commit()

    elif scenario == 2:
        # Scenario 2: Legacy Compensation & Salary Banding
        cur.execute("""
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                raw_salary TEXT NOT NULL,
                bonus_pct TEXT,
                dept_code TEXT
            );
        """)
        cur.executemany(
            "INSERT INTO employees (name, raw_salary, bonus_pct, dept_code) VALUES (?, ?, ?, ?);",
            [
                ("Dr. Jane Doe", "$120,000", "15%", "ENG"),
                ("John Smith, Jr.", "$85,000.50", "10%", "SALES"),
                ("Cher", "92000", None, "MKT"),
                ("Anonymous", "$0", "0%", None),
                ("Sam Gamgee", "$45,500", "5%", "DEV"),
            ],
        )
        conn.commit()

    return adapter


def render_sidebar():
    """Render key configuration, diagnostics, and parameters."""
    with st.sidebar:
        st.header("⚙️ Engine Configuration")

        # Load environment with UTF-8 BOM tolerance
        load_env_file()
        clean_key, key_src, masked_key = get_api_key_info(st.session_state.get("api_key_override"))

        st.subheader("🔑 API Key Status")
        if is_placeholder_key(clean_key):
            st.warning("⚠️ No valid key detected. Enter one below or update .env.")
        else:
            st.success(f"Key loaded from **{key_src}** ({masked_key})")

        key_input = st.text_input(
            "Nebius API Key (Override)",
            value=st.session_state.get("api_key_override", ""),
            type="password",
            help="Your key is only kept in session memory and never logged.",
        )
        if key_input != st.session_state.get("api_key_override"):
            st.session_state["api_key_override"] = key_input
            st.rerun()

        # Connection test button
        if st.button("🧪 Test Nebius Connection", width="stretch"):
            try:
                client = get_client(clean_key)
                fast, ultra, warns = resolve_nebius_models(client)
                st.success(f"Connected! Fast: `{fast}` | Ultra: `{ultra}`")
                for w in warns:
                    st.warning(w)
            except AgentError as ae:
                st.error(str(ae))
            except Exception as e:
                st.error(f"Connection test failed: {e}")

        st.markdown("---")
        st.subheader("🛡️ Safety Controls")
        st.session_state["fk_enabled"] = st.toggle(
            "App runs with Foreign Keys ON",
            value=st.session_state.get("fk_enabled", True),
            help="Enforces foreign key checks during baseline and migration audit.",
        )
        st.session_state["inject_traps"] = st.toggle(
            "Inject Adversarial Traps",
            value=st.session_state.get("inject_traps", True),
            help="Injects schema-driven edge cases (NULLs, Unicode, orphan keys) on clone.",
        )
        st.session_state["sandbox_timeout_s"] = st.slider(
            "Sandbox Base Timeout (s)",
            min_value=1.0,
            max_value=10.0,
            value=float(st.session_state.get("sandbox_timeout_s", 2.0)),
            step=0.5,
            help="Base timeout deadline for sandboxed execution (scales with row count up to 30s).",
        )
        st.session_state["max_attempts"] = st.number_input(
            "Max Attempts Budget",
            min_value=1,
            max_value=15,
            value=int(st.session_state.get("max_attempts", 6)),
            step=1,
            help="Maximum verification dry-run attempts.",
        )
        st.session_state["agent_wall_clock_timeout_s"] = st.number_input(
            "Wall Clock Budget (s)",
            min_value=30,
            max_value=1200,
            value=int(st.session_state.get("agent_wall_clock_timeout_s", 240)),
            step=30,
            help="Total wall-clock timeout budget for agent execution.",
        )

        st.markdown("---")
        st.caption("SQL Migration Dry-Run Time Machine v2.0")
        st.caption("Powered by Nebius Token Factory & NVIDIA Nemotron")


def render_connect_tab():
    """Tab 1: Connect to source database and introspect schema."""
    st.header("1. Connect to Database")
    st.write("Connect a real SQLite database or choose a pre-configured demo scenario. **The source database is opened read-only and is NEVER modified.**")

    source_type = st.radio(
        "Select Database Source",
        [
            "Built-in Scenario: Dept Normalization",
            "Built-in Scenario: Salary Banding",
            "Upload .sqlite / .db File",
            "Upload .sql Dump Script",
            "Local Database Filepath",
        ],
        horizontal=True,
    )

    if source_type == "Built-in Scenario: Dept Normalization":
        fp = "builtin:scenario_1"
        if st.session_state.get("source_fingerprint") != fp or st.session_state.get("adapter") is None:
            adapter = create_builtin_scenario_adapter(1)
            set_source(adapter, "Scenario 1", fp)
            st.session_state["target_table"] = "employees"
            st.session_state["goal"] = "Split full_name into first_name/last_name and add a foreign key department_code -> departments(dept_code); orphan codes and NULLs become NULL."

    elif source_type == "Built-in Scenario: Salary Banding":
        fp = "builtin:scenario_2"
        if st.session_state.get("source_fingerprint") != fp or st.session_state.get("adapter") is None:
            adapter = create_builtin_scenario_adapter(2)
            set_source(adapter, "Scenario 2", fp)
            st.session_state["target_table"] = "employees"
            st.session_state["goal"] = "Clean raw_salary by converting string amounts ($120,000) to integer cents in salary_cents, and compute salary_band."

    elif source_type == "Upload .sqlite / .db File":
        uploaded = st.file_uploader("Upload SQLite database (.sqlite, .db)", type=["sqlite", "db", "sqlite3"])
        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            fp = (uploaded.name, len(raw_bytes), hashlib.sha256(raw_bytes).hexdigest())
            if st.session_state.get("source_fingerprint") != fp or st.session_state.get("adapter") is None:
                adapter = SQLiteAdapter.from_bytes(raw_bytes)
                set_source(adapter, uploaded.name, fp)
                st.success(f"Loaded '{uploaded.name}' ({len(raw_bytes) / 1024:.1f} KB) into isolated sandbox memory.")

    elif source_type == "Upload .sql Dump Script":
        uploaded_sql = st.file_uploader("Upload SQL schema dump script (.sql)", type=["sql"])
        if uploaded_sql is not None:
            raw_bytes = uploaded_sql.getvalue()
            fp = (uploaded_sql.name, len(raw_bytes), hashlib.sha256(raw_bytes).hexdigest())
            if st.session_state.get("source_fingerprint") != fp or st.session_state.get("adapter") is None:
                dump_str = raw_bytes.decode("utf-8", errors="replace")
                adapter = SQLiteAdapter.from_sql_dump(dump_str)
                set_source(adapter, uploaded_sql.name, fp)
                st.success(f"Executed dump '{uploaded_sql.name}' into sandbox memory.")

    elif source_type == "Local Database Filepath":
        local_path = st.text_input("Absolute path to SQLite database on disk:", value="")
        if local_path and os.path.exists(local_path):
            mtime = os.path.getmtime(local_path)
            fp = (os.path.abspath(local_path), mtime)
            if st.button("Open Local DB"):
                adapter = SQLiteAdapter.from_file(local_path)
                set_source(adapter, os.path.basename(local_path), fp)
                st.success(f"Connected to '{local_path}' in read-only mode.")

    adapter = st.session_state.get("adapter")
    if adapter:
        schema = adapter.introspect()
        st.subheader(f"📊 Introspected Schema: {st.session_state.get('adapter_name')}")

        t_names = list(schema.tables.keys())
        if not t_names:
            st.warning("No tables found in this database.")
            return

        col1, col2 = st.columns([1, 2])
        with col1:
            prev_target = st.session_state.get("target_table")
            target_table = st.selectbox(
                "Primary Migration Target Table",
                t_names,
                index=t_names.index(prev_target) if prev_target in t_names else 0,
            )
            if target_table != prev_target:
                st.session_state["target_table"] = target_table
                st.session_state["spec"] = None
                st.session_state["spec_approved"] = False
                st.session_state["run_result"] = None

        with col2:
            tbl_info = schema.get_table(target_table)
            if tbl_info:
                st.metric("Total Rows in Target Table", f"{tbl_info.row_count:,}")

        # Show Schema Details
        with st.expander(f"View Columns & Constraints for `{target_table}`", expanded=True):
            cols_data = []
            for c in tbl_info.columns if tbl_info else []:
                cols_data.append({
                    "Column": c.name,
                    "Type": c.type,
                    "Not Null": c.notnull,
                    "Primary Key": c.pk,
                    "Default": c.default,
                })
            st.dataframe(cols_data, width="stretch")

            if tbl_info and tbl_info.foreign_keys:
                st.markdown("**Foreign Keys:**")
                st.json([fk.model_dump() for fk in tbl_info.foreign_keys])

            if tbl_info and tbl_info.indexes:
                st.markdown("**Indexes:**")
                st.json([idx.model_dump() for idx in tbl_info.indexes])

        # Sample rows with privacy masking
        with st.expander("🔍 Sample Rows (PII & Sensitive Values Masked)", expanded=False):
            clone = adapter.clone()
            try:
                cur = clone.cursor()
                cur.execute(f"SELECT * FROM \"{target_table}\" LIMIT 5")
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
                masked_samples = [
                    {cols[i]: mask_value(r[i]) for i in range(len(cols))}
                    for r in rows
                ]
                st.dataframe(masked_samples, width="stretch")
            finally:
                clone.close()


def render_plan_tab():
    """Tab 2: Formulate, inspect, and approve structured MigrationSpec."""
    st.header("2. Migration Planning & Spec Verification")
    st.write("Before generating any SQL, the AI creates a structured **MigrationSpec**. Inspect and approve the plan to guarantee expectations match reality.")

    adapter = st.session_state.get("adapter")
    if not adapter:
        st.warning("Please connect a database in Tab 1 first.")
        return

    goal_input = st.text_area(
        "Migration Goal Statement:",
        value=st.session_state.get("goal", ""),
        height=90,
        help="Describe what schema transformation should occur.",
    )
    if goal_input != st.session_state.get("goal"):
        st.session_state["goal"] = goal_input
        st.session_state["spec"] = None
        st.session_state["spec_approved"] = False
        st.session_state["run_result"] = None

    col1, col2, col3 = st.columns([1, 1, 1])
    generate_clicked = False
    with col1:
        if st.button("🤖 Generate Migration Spec (Nemotron Super)", width="stretch"):
            generate_clicked = True
    with col2:
        if st.session_state.get("spec") is not None:
            if st.button("🔄 Regenerate Migration Spec", width="stretch"):
                generate_clicked = True
    with col3:
        if st.button("🧪 Validate spec", width="stretch"):
            target_t = st.session_state.get("target_table", "employees")
            scenario_name = st.session_state.get("source_name", "")
            scenario_id = 1 if "Scenario 1" in scenario_name else (2 if "Scenario 2" in scenario_name else 1)
            try:
                ref_up, ref_down, ref_spec = get_builtin_scenario_reference(scenario_id)
                current_spec = st.session_state.get("spec") or ref_spec
                with st.spinner("Running reference rebuild satisfiability check in sandbox..."):
                    val_res = validate_spec_satisfiability(current_spec, adapter, ref_up, target_table=target_t)
                if val_res.passed:
                    st.success("✅ Spec validation PASSED: reference rebuild satisfied all invariants and oracle checks.")
                else:
                    st.error(f"❌ Spec validation FAILED: {'; '.join(val_res.failed_checks)}")
            except Exception as e:
                st.error(f"Spec validation error: {e}")

    if generate_clicked:
        st.session_state["run_result"] = None
        with st.spinner("Analyzing schema and drafting MigrationSpec..."):
            try:
                clean_key, _, _ = get_api_key_info(st.session_state.get("api_key_override"))
                client = get_client(clean_key)
                fast, _, warns = resolve_nebius_models(client)
                for w in warns:
                    st.warning(w)

                fp = st.session_state.get("source_fingerprint", "default")
                schema = get_cached_introspect(fp, adapter)
                traps = st.session_state.get("inject_traps", True)
                profiles = get_cached_profile(fp, traps)
                if profiles is None:
                    clone = adapter.clone()
                    try:
                        profiles = profile_database(clone, schema)
                        set_cached_profile(fp, traps, profiles)
                    except Exception as p_exc:
                        logger.debug(f"Could not compute profiles: {p_exc}")
                        profiles = {}
                    finally:
                        clone.close()

                target_t = st.session_state.get("target_table", "employees")
                spec_res = generate_spec(
                    client=client,
                    model=fast,
                    schema=schema,
                    goal=goal_input,
                    profiles=profiles,
                    target_table=target_t,
                )

                if spec_res.spec is not None:
                    st.session_state["spec"] = spec_res.spec
                    st.session_state["spec_approved"] = False
                    st.session_state["spec_source"] = spec_res.source
                    st.session_state["spec_model"] = spec_res.model
                    st.session_state["spec_latency_ms"] = spec_res.latency_ms
                    st.session_state["spec_goal_hash"] = hashlib.sha256(goal_input.encode()).hexdigest()[:8]
                    st.success("MigrationSpec generated successfully!")
                else:
                    st.session_state["spec"] = None
                    st.session_state["spec_approved"] = False
                    st.error(f"Failed to generate spec: {spec_res.error}")
            except Exception as exc:
                st.session_state["spec"] = None
                st.session_state["spec_approved"] = False
                st.error(f"Failed to generate spec: {exc}")

    spec: MigrationSpec | None = st.session_state.get("spec")
    if spec:
        target_t = st.session_state.get("target_table", "employees")
        if not spec.get_table_mappings(target_t):
            st.warning("⚠️ No column mappings for target table (no oracle: result will be UNVERIFIED).")

        st.subheader("📋 Generated MigrationSpec (JSON)")
        st.json(spec.model_dump())

        src = st.session_state.get("spec_source", "model")
        mdl = st.session_state.get("spec_model", "nemotron")
        lat = st.session_state.get("spec_latency_ms", 0.0)
        ghash = st.session_state.get("spec_goal_hash", "")
        st.caption(f"**Source**: `{src}` | **Model**: `{mdl}` | **Time**: `{round(lat)}ms` | **Goal Hash**: `{ghash}`")

        st.markdown("### User Approval Gate")
        approved = st.checkbox(
            "I have reviewed and approve this MigrationSpec for dry-run simulation.",
            value=st.session_state.get("spec_approved", False),
        )
        st.session_state["spec_approved"] = approved
        if approved:
            st.success("✅ MigrationSpec APPROVED. You may proceed to Tab 3: Run.")
        else:
            st.info("Please review the specification and check the approval box above.")


def render_run_tab():
    """Tab 3: Autonomous dry-run execution with live step trace."""
    st.header("3. Autonomous Dry-Run Simulation")
    st.write("The agent executes migration scripts strictly in an isolated in-memory RAM clone. Invariants and oracles are audited at every step.")

    adapter = st.session_state.get("adapter")
    if not adapter:
        st.warning("Please connect a database first.")
        return

    spec: MigrationSpec | None = st.session_state.get("spec")
    if not spec:
        st.warning("Please generate a MigrationSpec in Tab 2 before running.")
        return

    if not st.session_state.get("spec_approved", False):
        st.warning("⚠️ MigrationSpec requires user approval in Tab 2 before execution.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.session_state["is_naive_baseline"] = st.checkbox(
            "Simulate Naive Baseline First (Demo Mode)",
            value=st.session_state.get("is_naive_baseline", False),
            help="Generates an intentionally naive first-pass to demonstrate invariant catching and model escalation.",
        )

    # Launch agent run button
    can_run = st.session_state.get("spec_approved", False)
    if st.button("🚀 Start Autonomous Dry-Run Time Machine", disabled=not can_run, width="stretch"):
        status_container = st.status("Executing Autonomous Dry-Run Time Machine...", expanded=True)

        def on_progress(msg: str, data: dict):
            status_container.write(f"⏱️ **{msg}**")

        try:
            clean_key, _, _ = get_api_key_info(st.session_state.get("api_key_override"))
            target_t = st.session_state.get("target_table", "employees")
            goal_text = st.session_state.get("goal", "")

            # Inject traps if configured
            if st.session_state.get("inject_traps", True):
                setattr(adapter, "inject_traps", True)

            if hasattr(adapter, "settings") and "sandbox_timeout_s" in st.session_state:
                adapter.settings.sandbox_timeout_s = float(st.session_state["sandbox_timeout_s"])

            result = run_time_machine_agent(
                adapter=adapter,
                goal=goal_text,
                spec=spec,
                target_table=target_t,
                api_key_override=clean_key,
                max_steps=int(st.session_state.get("max_attempts", 6)),
                is_naive_baseline=st.session_state.get("is_naive_baseline", False),
                progress_callback=on_progress,
                wall_clock_timeout_s=float(st.session_state.get("agent_wall_clock_timeout_s", 240.0)),
            )

            st.session_state["run_result"] = result
            save_run_history(result, target_t, goal_text)

            if result.status in ("VERIFIED", "UNVERIFIED"):
                status_container.update(label=f"Dry-run Complete: {result.status}!", state="complete", expanded=False)
                st.success(f"Dry-Run Outcome: **{result.status}**")
            else:
                status_container.update(label=f"Dry-run Finished: {result.status}", state="error", expanded=True)
                st.error(f"Dry-Run Outcome: **{result.status}** ({result.stop_reason})")

        except AgentError as ae:
            status_container.update(label="Agent Error", state="error")
            st.error(f"Agent Execution Error: {ae}")
        except Exception as exc:
            status_container.update(label="System Error", state="error")
            st.error(f"Unexpected Execution Error: {exc}")

    # Display results if available in session_state
    res: AgentRunResult | None = st.session_state.get("run_result")
    if res:
        if res.model_warnings:
            for w in res.model_warnings:
                st.warning(w)
        st.subheader("Step-by-Step Execution Trace")
        for s in res.steps:
            with st.expander(f"Step {s.step}: [{s.model}] `{s.action}` ({round(s.latency_ms)}ms)"):
                st.json(s.details)


def render_result_tab():
    """Tab 4: Final verified outputs, code view, downloads, and full report."""
    st.header("4. Verified Migration Artifacts")

    res: AgentRunResult | None = st.session_state.get("run_result")
    if not res:
        st.info("No migration run completed yet. Execute a dry-run in Tab 3 to view results.")
        return

    target_t = st.session_state.get("target_table", "employees")
    goal_text = st.session_state.get("goal", "")

    # Top Status Banner
    if res.status == "VERIFIED":
        st.success("### 🟢 STATUS: VERIFIED (All Invariants & Reference Oracles Passed)")
    elif res.status == "UNVERIFIED":
        st.warning("### 🟡 STATUS: UNVERIFIED (Invariants Passed, Custom Oracle Needs Review)")
    else:
        st.error(f"### 🔴 STATUS: FAILED ({res.stop_reason})")

    # Invariant Summary Cards
    inv = res.invariant_result
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Baseline Rows", inv.before_count if inv else "n/a")
    with c2:
        st.metric("Migrated Rows", inv.after_count if inv and inv.after_count is not None else "n/a")
    with c3:
        if res.rollback_verified:
            rb_label = "✅ Verified"
        elif res.status == "FAILED":
            rb_label = "Skipped"
        elif res.rollback_classification == "LOSSY":
            rb_label = "⚠️ Lossy"
        elif res.rollback_error:
            rb_label = "❌ Failed"
        else:
            rb_label = "Not Verified"
        st.metric("Rollback Status", rb_label)
    with c4:
        st.metric("Est. Lock Duration", f"{res.lock_estimate.get('estimated_lock_sec', 0.0)}s")

    if res.status != "FAILED" and not res.rollback_verified and res.rollback_error:
        if res.rollback_classification == "LOSSY":
            st.warning(f"⚠️ **Rollback Roundtrip Warning (LOSSY):** {res.rollback_error}")
        else:
            st.warning(f"ℹ️ **Rollback Roundtrip Notice ({res.rollback_classification or 'NOT_VERIFIED'}):** {res.rollback_error}")

    # Attempts Table
    if getattr(res, "attempts", None):
        st.subheader("Attempts")
        attempts_rows = [
            {
                "Attempt #": a.attempt,
                "Model": a.model,
                "Passed": a.passed_count,
                "Failed": a.failed_count,
                "First Failed Check": a.first_failed_check or "-",
                "Sandbox Error": a.sandbox_error or "None",
                "Latency": f"{round(a.latency_ms)}ms",
                "Tokens": a.tokens,
            }
            for a in res.attempts
        ]
        st.dataframe(attempts_rows, use_container_width=True)

    # Transaction-wrapped Forward Script (up.sql)
    wrapped_up = wrap_transaction_with_prechecks(res.up_sql, target_t, res.spec)

    st.subheader("Forward Migration (`up.sql`)")
    st.code(wrapped_up, language="sql")

    st.subheader("Rollback Script (`down.sql`)")
    if res.rollback_verified:
        st.caption("✅ Roundtrip verified on disposable clone (zero data loss).")
    elif res.rollback_error:
        st.caption(f"⚠️ Verification notice: {res.rollback_error}")
    st.code(res.down_sql or "-- No rollback script provided", language="sql")

    # Full Markdown Report
    report_md = generate_markdown_report(res, target_t, goal_text)
    with st.expander("📄 View Full Markdown Verification Report", expanded=False):
        st.markdown(report_md)

    # Downloads & Multi-Format Exporters
    st.markdown("### 📥 Download & Export Migration Artifacts")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "Download up.sql",
            data=wrapped_up,
            file_name="up.sql",
            mime="text/plain",
            width="stretch",
        )
    with d2:
        st.download_button(
            "Download down.sql",
            data=res.down_sql,
            file_name="down.sql",
            mime="text/plain",
            width="stretch",
        )
    with d3:
        st.download_button(
            "Download Report (.md)",
            data=report_md,
            file_name="migration_report.md",
            mime="text/markdown",
            width="stretch",
        )
    with d4:
        alembic_code = export_alembic(res.up_sql, res.down_sql)
        st.download_button(
            "Export Alembic (.py)",
            data=alembic_code,
            file_name="alembic_migration.py",
            mime="text/x-python",
            width="stretch",
        )


def render_history_tab():
    """Tab 5: Local history of past dry-run executions (lazy loaded)."""
    st.header("5. Run History & Telemetry Logs")
    with st.expander("📜 Click to expand run history and telemetry logs", expanded=False):
        history = load_run_history(limit=25)
        if not history:
            st.info("No prior migration dry-run executions logged yet.")
            return

        for h in history:
            status_icon = "🟢" if h["status"] == "VERIFIED" else ("🟡" if h["status"] == "UNVERIFIED" else "🔴")
            with st.expander(f"{status_icon} Run #{h['id']} - {h['target_table']} ({h['timestamp']}) - {h['status']}"):
                st.write(f"**Goal:** {h['goal']}")
                st.write(f"**Latency:** {round(h['latency_ms'] / 1000, 2)}s | **Steps:** {h['steps_count']} | **Rollback:** {'Yes' if h['rollback_verified'] else 'No'}")
                st.code(h["up_sql"], language="sql")
                if h.get("report_json"):
                    try:
                        rep = json.loads(h["report_json"])
                        if rep.get("attempts"):
                            st.markdown("**Attempts & Raw Model Replies:**")
                            for att in rep["attempts"]:
                                att_num = att.get("attempt", 1)
                                att_model = att.get("model", "")
                                att_raw = (att.get("raw_reply") or "")[:1000]
                                with st.expander(f"Attempt #{att_num} ({att_model}) Raw Reply", expanded=False):
                                    st.code(att_raw or "(no raw reply)", language="text")
                    except Exception:
                        pass


def render_app():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="SQL Migration Dry-Run Time Machine",
        page_icon="⏱️",
        layout="wide",
    )
    init_session_state()
    render_sidebar()

    st.title("⏱️ SQL Migration Dry-Run Time Machine")
    st.caption("Autonomous Agentic SQL Schema Migrations on Disposable Clones with Spec-First Formal Verification")

    tabs = st.tabs([
        "1. Connect 🔌",
        "2. Plan 📋",
        "3. Run 🚀",
        "4. Result 🎯",
        "5. History 📜",
    ])

    with tabs[0]:
        render_connect_tab()
    with tabs[1]:
        render_plan_tab()
    with tabs[2]:
        render_run_tab()
    with tabs[3]:
        render_result_tab()
    with tabs[4]:
        render_history_tab()
