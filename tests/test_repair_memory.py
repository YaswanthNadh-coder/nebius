"""Tests for Item B: Repair Memory, AttemptHistory tracking, cumulative requirements, and conversation continuity."""

from unittest.mock import MagicMock

from timemachine.agent.loop import AttemptHistory, run_time_machine_agent
from timemachine.agent.prompts import format_previous_attempts_block
from timemachine.verify.invariants import InvariantResult
from timemachine.verify.spec import ColumnMapping, MigrationSpec


def test_attempt_history_structure_and_tracking():
    """AttemptHistory tracks attempt number, model, sql hash, approach summary, and failure types."""
    history = AttemptHistory()
    history.record_attempt(
        attempt_num=1,
        model="test-fast-model",
        up_sql="CREATE TABLE employees_new (id INT, salary INT NOT NULL);",
        inv_result=InvariantResult(
            status="FAILED",
            passed=False,
            failed_checks=["SQLITE EXECUTION CRASH: NOT NULL constraint failed: employees_new.salary"],
        ),
    )
    assert len(history.records) == 1
    rec = history.records[0]
    assert rec.attempt_num == 1
    assert rec.model == "test-fast-model"
    assert rec.sql_hash != ""
    assert any("salary" in t.lower() or "not_null" in t.lower() for t in rec.failed_check_types)
    assert len(rec.approach_summary) <= 200


def test_cumulative_requirements_and_reintroduced_warning():
    """If attempt 1 fails with X, attempt 2 fixes X but fails with Y:

    attempt 3 prompt contains BOTH X and Y requirements, and warns if X was reintroduced.
    """
    history = AttemptHistory()

    # Attempt 1: fails with NOT NULL on salary
    history.record_attempt(
        attempt_num=1,
        model="Super",
        up_sql="CREATE TABLE employees_new (id INT, salary INT NOT NULL);",
        inv_result=InvariantResult(
            status="FAILED",
            passed=False,
            failed_checks=["SQLITE EXECUTION CRASH: NOT NULL constraint failed: employees_new.salary"],
        ),
    )

    # Attempt 2: fixes salary, but mutates department
    history.record_attempt(
        attempt_num=2,
        model="Super",
        up_sql="CREATE TABLE employees_new (id INT, salary INT, department TEXT);",
        inv_result=InvariantResult(
            status="FAILED",
            passed=False,
            failed_checks=["UNTOUCHED_COLUMN_MUTATED: Column 'department' declared unchanged was modified."],
        ),
    )

    reqs = history.get_discovered_requirements()
    # Must contain both NOT NULL / constraint requirement and UNTOUCHED requirement
    req_text = " ".join(reqs).upper()
    assert "NOT NULL" in req_text or "SALARY" in req_text
    assert "UNTOUCHED" in req_text or "DEPARTMENT" in req_text

    # Attempt 3: reintroduces NOT NULL failure from attempt 1
    reintroduced = history.check_reintroduced_failures([
        "SQLITE EXECUTION CRASH: NOT NULL constraint failed: employees_new.salary"
    ])
    assert len(reintroduced) > 0
    assert any("salary" in r.lower() or "not null" in r.lower() for r in reintroduced)

    # Check formatted prompt block
    summary_block = format_previous_attempts_block(history)
    assert "Previous attempts" in summary_block
    assert "Requirements discovered so far" in summary_block


def test_fake_model_repair_conversation_continuity(monkeypatch):
    """Ensure conversation continues across repair attempts with system message once and cumulative feedback."""
    from timemachine.ui.app_view import create_builtin_scenario_adapter

    adapter = create_builtin_scenario_adapter(1)
    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="first_name", source_column="full_name", transform="split_name_first"),
                ColumnMapping(new_column="last_name", source_column="full_name", transform="split_name_last"),
            ]
        },
    )

    recorded_messages_history = []

    # Mock call_llm to record messages passed to each call
    def mock_call_llm(client, model, messages, **kwargs):
        recorded_messages_history.append(list(messages))
        # Attempt 1 returns SQL failing check X
        if len(recorded_messages_history) == 1:
            content = "```sql\n-- ==== UP ====\nCREATE TABLE employees_new (id INT PRIMARY KEY, first_name TEXT NOT NULL, salary INT NOT NULL);\n-- ==== DOWN ====\nCREATE TABLE employees_old (id INT);\n```"
        else:
            # Attempt 2 returns SQL failing check Y
            content = "```sql\n-- ==== UP ====\nCREATE TABLE employees_new (id INT PRIMARY KEY, first_name TEXT, last_name TEXT);\n-- ==== DOWN ====\nCREATE TABLE employees_old (id INT);\n```"
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = None
        return msg, MagicMock(total_tokens=100)

    monkeypatch.setattr("timemachine.agent.loop.call_llm", mock_call_llm)
    monkeypatch.setattr("timemachine.agent.loop.resolve_nebius_models", lambda client: ("fast-model", "ultra-model", []))

    run_time_machine_agent(
        adapter=adapter,
        goal="Split names",
        spec=spec,
        max_steps=2,
    )

    # Must have recorded 2 calls
    assert len(recorded_messages_history) >= 2
    second_call_messages = recorded_messages_history[1]

    # System message must be present
    assert any(m.get("role") == "system" for m in second_call_messages)
    # The second call must contain the failure feedback in the user turn
    user_turns = [m for m in second_call_messages if m.get("role") == "user"]
    assert len(user_turns) >= 2
    last_user_turn = user_turns[-1]["content"]
    assert "Previous attempts" in last_user_turn or "Requirements discovered so far" in last_user_turn
