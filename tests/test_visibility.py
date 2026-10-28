"""Tests for Item A: Visibility (progress logging, distinct repair messages, Attempts table, history)."""

import json
from unittest.mock import MagicMock, patch

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.agent.loop import AgentRunResult, AttemptRecord, run_time_machine_agent
from timemachine.report.history import load_run_history, save_run_history
from timemachine.verify.spec import MigrationSpec


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, content=None, tool_calls=None, finish_reason="stop"):
        self.message = FakeMessage(content, tool_calls)
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, content=None, tool_calls=None, finish_reason="stop"):
        self.choices = [FakeChoice(content, tool_calls, finish_reason)]
        self.usage = MagicMock(total_tokens=150)


def _setup_adapter_and_spec():
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.execute("INSERT INTO employees VALUES (1, 'Alice');")
    conn.commit()

    spec = MigrationSpec(
        goal="Add email column",
        affected_tables=["employees"],
    )
    return adapter, spec


def test_progress_log_appends_first_failed_check():
    """In live progress log, append first failed check (<=160 chars) to Verification outcome: FAILED (passed N, failed M)."""
    adapter, spec = _setup_adapter_and_spec()

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]
    # Return SQL that causes syntax error
    client.chat.completions.create.return_value = FakeResponse(
        content="```sql\nINVALID SQL SYNTAX HERE;\n```"
    )

    notifications: list[str] = []

    def on_progress(msg: str, data: dict):
        notifications.append(msg)

    with patch("timemachine.agent.loop.get_client", return_value=client):
        run_time_machine_agent(
            adapter=adapter,
            goal="Add email",
            spec=spec,
            target_table="employees",
            max_steps=1,
            progress_callback=on_progress,
        )

    # Find the verification outcome line
    outcome_logs = [n for n in notifications if "Verification outcome: FAILED" in n]
    assert len(outcome_logs) > 0, f"Expected Verification outcome: FAILED in logs, got {notifications}"
    first_log = outcome_logs[0]
    # Must contain passed N, failed M
    assert "passed" in first_log and "failed" in first_log
    # Must contain the failure details (syntax error)
    assert "INVALID" in first_log or "syntax" in first_log.lower() or "error" in first_log.lower()


def test_repair_notification_distinct_messages():
    """Distinct repair messages: 'Attempt 2/6 (Ultra): repairing <first failed check>' and 'No SQL in reply...'."""
    adapter, spec = _setup_adapter_and_spec()

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # Sequence:
    # 1. First call: returns invalid SQL (fails verification)
    # 2. Second call: returns no SQL text at all (triggers no-SQL reply)
    # 3. Third call: returns valid SQL
    client.chat.completions.create.side_effect = [
        FakeResponse(content="```sql\nINVALID SYNTAX;\n```"),
        FakeResponse(content="I am thinking about what to do without SQL."),
        FakeResponse(content="```sql\nALTER TABLE employees ADD COLUMN email TEXT;\n-- DOWN\nALTER TABLE employees DROP COLUMN email;\n```"),
    ]

    notifications: list[str] = []

    def on_progress(msg: str, data: dict):
        notifications.append(msg)

    with patch("timemachine.agent.loop.get_client", return_value=client):
        run_time_machine_agent(
            adapter=adapter,
            goal="Add email",
            spec=spec,
            target_table="employees",
            max_steps=4,
            progress_callback=on_progress,
        )

    # Check for distinct attempt/repair notification
    repair_logs = [n for n in notifications if "repairing" in n]
    assert len(repair_logs) > 0, f"Expected 'repairing' in notifications: {notifications}"
    assert any("Attempt" in r and ("Ultra" in r or "Super" in r) for r in repair_logs), f"Expected 'Attempt X/Y (Model): repairing ...', got {repair_logs}"

    # Check for distinct no-SQL retry notification
    nosql_logs = [n for n in notifications if "No SQL in reply" in n]
    assert len(nosql_logs) > 0, f"Expected 'No SQL in reply' in notifications: {notifications}"
    assert any("retry" in n for n in nosql_logs), f"Expected retry count in no-SQL notification: {nosql_logs}"


def test_agent_run_result_tracks_attempts_table():
    """Result has attempts table tracking each attempt with model, passed, failed, first_failed, latency, tokens, raw_reply."""
    adapter, spec = _setup_adapter_and_spec()

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]
    client.chat.completions.create.side_effect = [
        FakeResponse(content="```sql\nINVALID SQL;\n```"),
        FakeResponse(content="```sql\nALTER TABLE employees ADD COLUMN email TEXT;\n-- DOWN\nALTER TABLE employees DROP COLUMN email;\n```"),
    ]

    with patch("timemachine.agent.loop.get_client", return_value=client):
        res = run_time_machine_agent(
            adapter=adapter,
            goal="Add email",
            spec=spec,
            target_table="employees",
            max_steps=3,
        )

    assert hasattr(res, "attempts")
    assert len(res.attempts) >= 2
    att1 = res.attempts[0]
    assert att1.attempt == 1
    assert att1.failed_count > 0
    assert att1.first_failed_check != ""
    assert att1.raw_reply != ""
    assert len(att1.raw_reply) <= 1000

    att2 = res.attempts[1]
    assert att2.attempt == 2
    assert att2.passed_count > 0


def test_history_saves_raw_reply_per_attempt(tmp_path):
    """History persists attempts with raw model reply (first 1000 chars)."""
    db_file = str(tmp_path / "test_runs.db")

    res = AgentRunResult(
        status="VERIFIED",
        up_sql="ALTER TABLE employees ADD COLUMN email TEXT;",
        down_sql="ALTER TABLE employees DROP COLUMN email;",
        attempts=[
            AttemptRecord(
                attempt=1,
                model="nvidia/nemotron-3-super-120b-a12b",
                passed_count=0,
                failed_count=1,
                first_failed_check="near 'INVALID': syntax error",
                latency_ms=120.0,
                tokens=50,
                raw_reply="Here is invalid sql: ```sql\nINVALID\n```",
            ),
            AttemptRecord(
                attempt=2,
                model="nvidia/Nemotron-3-Ultra-550b-a55b",
                passed_count=5,
                failed_count=0,
                first_failed_check="",
                latency_ms=250.0,
                tokens=120,
                raw_reply="Here is the fix: ```sql\nALTER TABLE...\n```",
            ),
        ],
    )

    row_id = save_run_history(res, "employees", "Add email", db_path=db_file)
    assert row_id > 0

    history = load_run_history(limit=5, db_path=db_file)
    assert len(history) == 1
    entry = history[0]
    report_data = json.loads(entry["report_json"])
    assert "attempts" in report_data
    attempts = report_data["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["raw_reply"].startswith("Here is invalid sql")
    assert attempts[1]["raw_reply"].startswith("Here is the fix")
