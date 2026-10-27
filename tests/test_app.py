"""Streamlit AppTest suite verifying widget state persistence and zero uncaught exceptions."""

import os

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
    monkeypatch.setenv("NEBIUS_API_KEY", "test_key_fake_123456")
    monkeypatch.setenv("NEBIUS_FAST_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("NEBIUS_ULTRA_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")

    class FakeMessage:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class FakeChoice:
        def __init__(self, content=None, tool_calls=None):
            self.message = FakeMessage(content, tool_calls)
            self.finish_reason = "stop"

    class FakeResponse:
        def __init__(self, content=None, tool_calls=None):
            self.choices = [FakeChoice(content, tool_calls)]
            self.usage = None

    def fake_create(self, *args, **kwargs):
        return FakeResponse(content="```sql\nALTER TABLE employees ADD COLUMN test_col TEXT DEFAULT 'ok';\n```")

    import openai.resources.chat.completions
    monkeypatch.setattr(openai.resources.chat.completions.Completions, "create", fake_create)

    app_path = os.path.abspath("app.py")
    at = AppTest.from_file(app_path, default_timeout=15.0).run()

    # Approve spec and set naive baseline false
    from timemachine.verify.spec import MigrationSpec

    at.session_state["spec"] = MigrationSpec(affected_tables=["employees"], goal="test")
    at.session_state["spec_approved"] = True
    at.session_state["inject_traps"] = True
    at.run()

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


def test_app_create_spec_approve_and_run_later(monkeypatch):
    """AppTest creates a spec in one run, approves it, and clicks Run in a LATER run."""
    import json
    monkeypatch.setenv("NEBIUS_API_KEY", "test_key_fake_123456")
    monkeypatch.setenv("NEBIUS_FAST_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("NEBIUS_ULTRA_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")

    class FakeMessage:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class FakeChoice:
        def __init__(self, content=None, tool_calls=None):
            self.message = FakeMessage(content, tool_calls)
            self.finish_reason = "stop"

    class FakeResponse:
        def __init__(self, content=None, tool_calls=None):
            self.choices = [FakeChoice(content, tool_calls)]
            self.usage = None

    valid_spec_json = json.dumps({
        "goal": "Normalize department",
        "affected_tables": ["employees"],
    })

    def fake_create(self, messages, **kwargs):
        user_msg = messages[-1]["content"] if messages else ""
        if "Target JSON Schema" in user_msg or "output a structured MigrationSpec JSON" in user_msg:
            return FakeResponse(content=f"```json\n{valid_spec_json}\n```")
        return FakeResponse(content="```sql\nALTER TABLE employees ADD COLUMN test_col TEXT DEFAULT 'ok';\n```")

    import openai.resources.chat.completions
    monkeypatch.setattr(openai.resources.chat.completions.Completions, "create", fake_create)

    app_path = os.path.abspath("app.py")
    at = AppTest.from_file(app_path, default_timeout=15.0).run()

    # Step 1: Click "Generate Migration Spec"
    all_buttons = list(at.button)
    if len(at.tabs) > 1:
        all_buttons.extend(list(at.tabs[1].button))
    gen_btn = next((b for b in all_buttons if "Generate Migration Spec" in b.label), None)
    assert gen_btn is not None
    gen_btn.click().run()

    assert not at.exception
    assert at.session_state.get("spec") is not None

    # Step 2: Approve the spec in a subsequent run
    all_cbs = list(at.checkbox)
    if len(at.tabs) > 1:
        all_cbs.extend(list(at.tabs[1].checkbox))
    appr_cb = next((c for c in all_cbs if "approve this MigrationSpec" in c.label), None)
    assert appr_cb is not None
    appr_cb.check().run()
    assert at.session_state.get("spec_approved") is True

    # Step 3: Click Run in a LATER run
    all_run_buttons = list(at.button)
    if len(at.tabs) > 2:
        all_run_buttons.extend(list(at.tabs[2].button))
    run_btn = next((b for b in all_run_buttons if "Start Autonomous Dry-Run" in b.label), None)
    assert run_btn is not None
    assert not run_btn.disabled
    run_btn.click().run()

    assert not at.exception
    error_texts = [e.value for e in at.error]
    assert len(error_texts) == 0, f"Unexpected st.error in app: {error_texts}"
    assert at.session_state.get("run_result") is not None
    assert at.session_state["run_result"].status in ("VERIFIED", "UNVERIFIED")


def test_app_switching_scenario_clears_spec():
    """Switching database scenario in Connect tab clears spec, spec_approved, and run_result."""
    app_path = os.path.abspath("app.py")
    at = AppTest.from_file(app_path, default_timeout=15.0).run()

    from timemachine.verify.spec import MigrationSpec
    at.session_state["spec"] = MigrationSpec(affected_tables=["employees"], goal="old goal")
    at.session_state["spec_approved"] = True
    at.run()

    # Switch radio to Scenario 2
    radios = list(at.radio)
    if len(at.tabs) > 0:
        radios.extend(list(at.tabs[0].radio))
    src_radio = next((r for r in radios if "Select Database Source" in r.label), None)
    assert src_radio is not None
    src_radio.set_value("Built-in Scenario: Salary Banding").run()

    assert at.session_state.get("spec") is None
    assert at.session_state.get("spec_approved") is False
    assert at.session_state.get("run_result") is None


def test_app_failed_generation_leaves_no_stale_spec(monkeypatch):
    """When spec generation fails, session_state['spec'] is cleared, not showing stale spec."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test_key_fake_123456")
    monkeypatch.setenv("NEBIUS_FAST_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("NEBIUS_ULTRA_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")

    class FakeChoice:
        def __init__(self):
            self.message = type("Msg", (), {"content": "not json at all", "tool_calls": None})()
            self.finish_reason = "stop"

    class FakeResponse:
        def __init__(self):
            self.choices = [FakeChoice()]
            self.usage = None

    import openai.resources.chat.completions
    monkeypatch.setattr(openai.resources.chat.completions.Completions, "create", lambda self, *args, **kwargs: FakeResponse())

    app_path = os.path.abspath("app.py")
    at = AppTest.from_file(app_path, default_timeout=15.0).run()

    from timemachine.verify.spec import MigrationSpec
    at.session_state["spec"] = MigrationSpec(affected_tables=["employees"], goal="stale goal")
    at.run()

    all_buttons = list(at.button)
    if len(at.tabs) > 1:
        all_buttons.extend(list(at.tabs[1].button))
    gen_btn = next((b for b in all_buttons if "Generate Migration Spec" in b.label), None)
    assert gen_btn is not None
    gen_btn.click().run()

    # Spec must be None after failure
    assert at.session_state.get("spec") is None
    assert len(at.error) > 0


def test_app_run_dry_run_with_naive_baseline_checked(monkeypatch):
    """Verify that clicking Start Autonomous Dry-Run with naive baseline checkbox ON executes cleanly."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test_key_fake_123456")
    monkeypatch.setenv("NEBIUS_FAST_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("NEBIUS_ULTRA_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")

    class FakeMessage:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class FakeChoice:
        def __init__(self, content=None, tool_calls=None):
            self.message = FakeMessage(content, tool_calls)
            self.finish_reason = "stop"

    class FakeResponse:
        def __init__(self, content=None, tool_calls=None):
            self.choices = [FakeChoice(content, tool_calls)]
            self.usage = None

    def fake_create(self, *args, **kwargs):
        return FakeResponse(content="```sql\nALTER TABLE employees ADD COLUMN test_col TEXT DEFAULT 'ok';\n```")

    import openai.resources.chat.completions
    monkeypatch.setattr(openai.resources.chat.completions.Completions, "create", fake_create)

    app_path = os.path.abspath("app.py")
    at = AppTest.from_file(app_path, default_timeout=15.0).run()

    from timemachine.verify.spec import MigrationSpec

    at.session_state["spec"] = MigrationSpec(affected_tables=["employees"], goal="test")
    at.session_state["spec_approved"] = True
    at.session_state["is_naive_baseline"] = True
    at.run()

    all_buttons = list(at.button)
    if len(at.tabs) > 2:
        all_buttons.extend(list(at.tabs[2].button))

    run_btn = next((b for b in all_buttons if "Start Autonomous Dry-Run" in b.label), None)
    assert run_btn is not None
    run_btn.click().run()

    assert not at.exception
    error_texts = [e.value for e in at.error]
    assert not any("Unexpected Execution Error" in t for t in error_texts)
    assert at.session_state.get("run_result") is not None
    assert at.session_state["run_result"].status in ("VERIFIED", "UNVERIFIED", "FAILED")


def test_app_upload_db_lifecycle_persists_spec_and_reupload_resets(monkeypatch, tmp_path):
    """Test uploaded .db: Generate -> Approve -> Run with spec persisting, and re-upload resetting."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test_key_fake_123456")
    monkeypatch.setenv("NEBIUS_FAST_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("NEBIUS_ULTRA_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")

    import sqlite3

    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT);")
    conn.execute("INSERT INTO employees VALUES (1, 'Alice');")
    conn.commit()
    conn.close()
    db_bytes = db_file.read_bytes()

    db_file2 = tmp_path / "test2.db"
    conn2 = sqlite3.connect(str(db_file2))
    conn2.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, amount REAL);")
    conn2.commit()
    conn2.close()
    db_bytes2 = db_file2.read_bytes()

    class FakeMessage:
        def __init__(self, content=None):
            self.content = content
            self.tool_calls = None

    class FakeChoice:
        def __init__(self, content=None):
            self.message = FakeMessage(content)
            self.finish_reason = "stop"

    class FakeResponse:
        def __init__(self, content=None):
            self.choices = [FakeChoice(content)]
            self.usage = None

    def fake_create(self, *args, **kwargs):
        messages = kwargs.get("messages", [])
        prompt_text = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        if "Target JSON Schema" in prompt_text or "MigrationSpec" in prompt_text:
            return FakeResponse(
                content="```json\n"
                "{\n"
                '  "affected_tables": ["employees"],\n'
                '  "goal": "Test goal",\n'
                '  "column_mappings": {"employees": [{"new_column": "name", "source_column": "name"}]},\n'
                '  "row_policy": "preserve_all"\n'
                "}\n"
                "```"
            )
        return FakeResponse(
            content="```sql\n"
            "ALTER TABLE employees ADD COLUMN first_name TEXT DEFAULT 'ok';\n"
            "```"
        )

    import openai.resources.chat.completions
    monkeypatch.setattr(openai.resources.chat.completions.Completions, "create", fake_create)

    app_path = os.path.abspath("app.py")
    at = AppTest.from_file(app_path, default_timeout=15.0).run()

    # Select upload .db in radio
    at.radio[0].set_value("Upload .sqlite / .db File").run()

    # Upload file
    at.file_uploader[0].upload("test.db", db_bytes).run()
    assert at.session_state.get("adapter") is not None

    # Step 1: Click Generate
    gen_btn = next((b for b in at.button if "Generate Migration Spec" in b.label), None)
    assert gen_btn is not None
    gen_btn.click().run()

    assert at.session_state.get("spec") is not None

    # Step 2: Approve the spec
    appr_cb = next((c for c in at.checkbox if "approve this MigrationSpec" in c.label), None)
    assert appr_cb is not None
    appr_cb.check().run()

    # Spec must NOT be wiped by the rerun!
    assert at.session_state.get("spec") is not None, "spec was wiped on approval rerun!"
    assert at.session_state.get("spec_approved") is True, "spec_approved was reset on rerun!"

    # Step 3: Run the simulation in a later run
    all_buttons = list(at.button)
    if len(at.tabs) > 2:
        all_buttons.extend(list(at.tabs[2].button))
    run_btn = next((b for b in all_buttons if "Start Autonomous Dry-Run" in b.label), None)
    assert run_btn is not None
    run_btn.click().run()

    assert not at.exception
    assert at.session_state.get("run_result") is not None

    # Step 4: Re-uploading a different file resets the state
    at.file_uploader[0].upload("test2.db", db_bytes2).run()
    assert at.session_state.get("spec") is None, "re-upload did not reset spec!"
    assert at.session_state.get("spec_approved") is False, "re-upload did not reset spec_approved!"
    assert at.session_state.get("run_result") is None, "re-upload did not reset run_result!"


def test_app_upload_sql_lifecycle_persists_spec_and_reupload_resets(monkeypatch):
    """Test uploaded .sql: Generate -> Approve -> Run with spec persisting, and re-upload resetting."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test_key_fake_123456")
    monkeypatch.setenv("NEBIUS_FAST_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("NEBIUS_ULTRA_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")

    sql_bytes = b"CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT);\nINSERT INTO employees VALUES (1, 'Alice');\n"
    sql_bytes2 = b"CREATE TABLE products (id INTEGER PRIMARY KEY, price REAL);\n"

    class FakeMessage:
        def __init__(self, content=None):
            self.content = content
            self.tool_calls = None

    class FakeChoice:
        def __init__(self, content=None):
            self.message = FakeMessage(content)
            self.finish_reason = "stop"

    class FakeResponse:
        def __init__(self, content=None):
            self.choices = [FakeChoice(content)]
            self.usage = None

    def fake_create(self, *args, **kwargs):
        messages = kwargs.get("messages", [])
        prompt_text = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        if "Target JSON Schema" in prompt_text or "MigrationSpec" in prompt_text:
            return FakeResponse(
                content="```json\n"
                "{\n"
                '  "affected_tables": ["employees"],\n'
                '  "goal": "Test goal",\n'
                '  "column_mappings": {"employees": [{"new_column": "name", "source_column": "name"}]},\n'
                '  "row_policy": "preserve_all"\n'
                "}\n"
                "```"
            )
        return FakeResponse(
            content="```sql\n"
            "ALTER TABLE employees ADD COLUMN first_name TEXT DEFAULT 'ok';\n"
            "```"
        )

    import openai.resources.chat.completions
    monkeypatch.setattr(openai.resources.chat.completions.Completions, "create", fake_create)

    app_path = os.path.abspath("app.py")
    at = AppTest.from_file(app_path, default_timeout=15.0).run()

    # Select upload .sql in radio
    at.radio[0].set_value("Upload .sql Dump Script").run()

    # Upload file
    at.file_uploader[0].upload("dump.sql", sql_bytes).run()
    assert at.session_state.get("adapter") is not None

    # Step 1: Click Generate
    gen_btn = next((b for b in at.button if "Generate Migration Spec" in b.label), None)
    assert gen_btn is not None
    gen_btn.click().run()

    assert at.session_state.get("spec") is not None

    # Step 2: Approve the spec
    appr_cb = next((c for c in at.checkbox if "approve this MigrationSpec" in c.label), None)
    assert appr_cb is not None
    appr_cb.check().run()

    # Spec must NOT be wiped by the rerun!
    assert at.session_state.get("spec") is not None, "spec was wiped on approval rerun!"
    assert at.session_state.get("spec_approved") is True, "spec_approved was reset on rerun!"

    # Step 3: Run the simulation in a later run
    all_buttons = list(at.button)
    if len(at.tabs) > 2:
        all_buttons.extend(list(at.tabs[2].button))
    run_btn = next((b for b in all_buttons if "Start Autonomous Dry-Run" in b.label), None)
    assert run_btn is not None
    run_btn.click().run()

    assert not at.exception
    assert at.session_state.get("run_result") is not None

    # Step 4: Re-uploading a different file resets the state
    at.file_uploader[0].upload("dump2.sql", sql_bytes2).run()
    assert at.session_state.get("spec") is None, "re-upload did not reset spec!"
    assert at.session_state.get("spec_approved") is False, "re-upload did not reset spec_approved!"
    assert at.session_state.get("run_result") is None, "re-upload did not reset run_result!"


def test_app_local_path_lifecycle_persists_spec_and_run(monkeypatch, tmp_path):
    """Test local path flow: select local path -> Open Local DB -> Generate -> Approve -> Run."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test_key_fake_123456")
    monkeypatch.setenv("NEBIUS_FAST_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("NEBIUS_ULTRA_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")

    import sqlite3

    db_file = tmp_path / "local_test.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT);")
    conn.execute("INSERT INTO employees VALUES (1, 'Alice');")
    conn.commit()
    conn.close()

    class FakeMessage:
        def __init__(self, content=None):
            self.content = content
            self.tool_calls = None

    class FakeChoice:
        def __init__(self, content=None):
            self.message = FakeMessage(content)
            self.finish_reason = "stop"

    class FakeResponse:
        def __init__(self, content=None):
            self.choices = [FakeChoice(content)]
            self.usage = None

    def fake_create(self, *args, **kwargs):
        messages = kwargs.get("messages", [])
        prompt_text = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        if "Target JSON Schema" in prompt_text or "MigrationSpec" in prompt_text:
            return FakeResponse(
                content="```json\n"
                "{\n"
                '  "affected_tables": ["employees"],\n'
                '  "goal": "Test goal",\n'
                '  "column_mappings": {"employees": [{"new_column": "name", "source_column": "name"}]},\n'
                '  "row_policy": "preserve_all"\n'
                "}\n"
                "```"
            )
        return FakeResponse(
            content="```sql\n"
            "ALTER TABLE employees ADD COLUMN first_name TEXT DEFAULT 'ok';\n"
            "```"
        )

    import openai.resources.chat.completions
    monkeypatch.setattr(openai.resources.chat.completions.Completions, "create", fake_create)

    app_path = os.path.abspath("app.py")
    at = AppTest.from_file(app_path, default_timeout=15.0).run()

    # Select Local Database Filepath in radio
    at.radio[0].set_value("Local Database Filepath").run()

    # Enter path and click Open Local DB
    # Find text input for path
    path_input = next((t for t in at.text_input if "Absolute path" in t.label), None)
    assert path_input is not None
    path_input.set_value(str(db_file)).run()

    open_btn = next((b for b in at.button if "Open Local DB" in b.label), None)
    assert open_btn is not None
    open_btn.click().run()

    assert at.session_state.get("adapter") is not None

    # Step 1: Click Generate
    gen_btn = next((b for b in at.button if "Generate Migration Spec" in b.label), None)
    assert gen_btn is not None
    gen_btn.click().run()

    assert at.session_state.get("spec") is not None

    # Step 2: Approve the spec
    appr_cb = next((c for c in at.checkbox if "approve this MigrationSpec" in c.label), None)
    assert appr_cb is not None
    appr_cb.check().run()

    assert at.session_state.get("spec_approved") is True

    # Step 3: Run the simulation in a later run
    all_buttons = list(at.button)
    if len(at.tabs) > 2:
        all_buttons.extend(list(at.tabs[2].button))
    run_btn = next((b for b in all_buttons if "Start Autonomous Dry-Run" in b.label), None)
    assert run_btn is not None
    run_btn.click().run()

    assert not at.exception
    error_texts = [e.value for e in at.error]
    assert not any("Unexpected Execution Error" in t for t in error_texts)
    assert at.session_state.get("run_result") is not None





