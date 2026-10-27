"""Streamlit AppTest suite verifying widget state persistence and zero uncaught exceptions."""

import os

import pytest
from streamlit.testing.v1 import AppTest


def test_app_initial_render_zero_exceptions():
    """Verify that app.py boots cleanly without any uncaught exceptions or tracebacks."""
    app_path = os.path.abspath("app.py")
    at = AppTest.from_file(app_path, default_timeout=15.0).run()
    assert not at.exception, f"App threw exception during boot: {at.exception}"
    assert len(at.tabs) == 5


def test_app_results_persist_across_widget_changes():
    """Verify that touching sidebar toggles and radio buttons does not wipe session state results."""
    app_path = os.path.abspath("app.py")
    at = AppTest.from_file(app_path, default_timeout=15.0).run()

    # Pre-populate dummy dry-run results into session state
    from timemachine.agent.loop import AgentRunResult
    from timemachine.verify.invariants import InvariantResult

    dummy_result = AgentRunResult(
        status="VERIFIED",
        up_sql="ALTER TABLE employees ADD COLUMN test_col TEXT;",
        down_sql="ALTER TABLE employees DROP COLUMN test_col;",
        rollback_verified=True,
        invariant_result=InvariantResult(
            status="VERIFIED",
            passed=True,
            before_count=100,
            after_count=100,
            elapsed_ms=1.2,
        ),
    )

    at.session_state["run_result"] = dummy_result
    at.session_state["spec_approved"] = True

    # Touch a widget: toggle foreign keys or traps
    at.toggle[0].set_value(False).run()

    assert not at.exception, "Exception raised after widget interaction"
    # Verify results still persist in session state
    assert at.session_state.get("run_result") is not None
    assert at.session_state["run_result"].status == "VERIFIED"
    assert at.session_state["spec_approved"] is True


def test_app_run_dry_run_simulation(monkeypatch):
    """Verify that clicking Start Autonomous Dry-Run Time Machine executes cleanly without system error."""
    app_path = os.path.abspath("app.py")
    at = AppTest.from_file(app_path, default_timeout=15.0).run()

    # Approve spec and set naive baseline false
    from timemachine.verify.spec import MigrationSpec

    at.session_state["spec"] = MigrationSpec(affected_tables=["employees"], goal="test")
    at.session_state["spec_approved"] = True
    at.session_state["inject_traps"] = True
    at.run()

    # Mock call_llm so dry run completes with clean SQL
    import timemachine.agent.loop as agent_loop

    def fake_call_llm(client, model, prompt, **kwargs):
        return ("ALTER TABLE employees ADD COLUMN test_col TEXT DEFAULT 'ok';", None)

    monkeypatch.setattr(agent_loop, "call_llm", fake_call_llm)

    # Find the Start Autonomous Dry-Run Time Machine button (may be in tabs[2] or global at.button)
    all_buttons = list(at.button)
    if len(at.tabs) > 2:
        all_buttons.extend(list(at.tabs[2].button))

    run_btn = next((b for b in all_buttons if "Start Autonomous Dry-Run" in b.label), None)
    assert run_btn is not None, f"Dry run button not found in {[b.label for b in all_buttons]}"
    assert not run_btn.disabled, "Run button should be enabled when spec_approved=True"

    run_btn.click().run()

    assert not at.exception, f"App threw uncaught exception during dry run: {at.exception}"
    # Verify no system error was displayed
    error_texts = [e.value for e in at.error]
    assert not any("Unexpected Execution Error" in t for t in error_texts), f"Found Unexpected Execution Error: {error_texts}"
    assert at.session_state.get("run_result") is not None
    assert at.session_state["run_result"].status in ("VERIFIED", "UNVERIFIED")
