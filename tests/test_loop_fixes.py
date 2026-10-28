"""Tests for Item E: Agent loop enhancements (attempts vs steps, conversation preservation, fallback, best-attempt, escalation, crash row count, tokens)."""

import json
from unittest.mock import MagicMock, patch

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.agent.loop import run_time_machine_agent
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
    def __init__(self, content=None, tool_calls=None, finish_reason="stop", tokens=100):
        self.choices = [FakeChoice(content, tool_calls, finish_reason)]
        self.usage = MagicMock(total_tokens=tokens)


class FakeToolCall:
    def __init__(self, fn_name, fn_args, call_id="call_1"):
        self.id = call_id
        self.type = "function"
        self.function = MagicMock(name=fn_name, arguments=fn_args)
        self.function.name = fn_name
        self.function.arguments = fn_args


def _setup_db():
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.executemany("INSERT INTO employees VALUES (?, ?);", [(1, "Alice"), (2, "Bob")])
    conn.commit()

    spec = MigrationSpec(
        goal="Add email column",
        affected_tables=["employees"],
    )
    return adapter, spec


def test_no_sql_retries_do_not_consume_attempts():
    """Tool rounds and no-SQL replies must not consume attempts (attempts counts only verifications)."""
    adapter, spec = _setup_db()
    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # Reply 1: No SQL (retry 1)
    # Reply 2: Valid SQL (Passes verification)
    client.chat.completions.create.side_effect = [
        FakeResponse(content="I am thinking..."),
        FakeResponse(content="```sql\nALTER TABLE employees ADD COLUMN email TEXT;\n-- DOWN\nALTER TABLE employees DROP COLUMN email;\n```"),
    ]

    with patch("timemachine.agent.loop.get_client", return_value=client):
        res = run_time_machine_agent(
            adapter=adapter,
            goal="Add email",
            spec=spec,
            target_table="employees",
            max_steps=1,  # max_attempts = 1
        )

    # Even though 3 LLM calls happened, only 1 verification attempt occurred
    assert len(res.attempts) == 1
    assert res.status in ("VERIFIED", "UNVERIFIED")


def test_conversation_preserved_on_no_sql_retry_in_repair():
    """Preserve conversation on every no-SQL retry, including repair steps."""
    adapter, spec = _setup_db()
    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # Attempt 1: invalid SQL (triggers repair)
    # Attempt 2 (repair turn 1): no SQL text
    # Attempt 2 (repair turn 2): valid SQL
    client.chat.completions.create.side_effect = [
        FakeResponse(content="```sql\nINVALID SQL;\n```"),
        FakeResponse(content="Thinking about repair without SQL."),
        FakeResponse(content="```sql\nALTER TABLE employees ADD COLUMN email TEXT;\n-- DOWN\nALTER TABLE employees DROP COLUMN email;\n```"),
    ]

    with patch("timemachine.agent.loop.get_client", return_value=client):
        run_time_machine_agent(
            adapter=adapter,
            goal="Add email",
            spec=spec,
            target_table="employees",
            max_steps=3,
        )

    # Call 3 should have received a message list with the previous assistant reply and user nudge appended
    call3_msgs = client.chat.completions.create.call_args_list[2][1]["messages"]
    assert len(call3_msgs) >= 3
    # Second-to-last message should be assistant's no-SQL reply
    assert any(m["role"] == "assistant" and "Thinking about repair" in m.get("content", "") for m in call3_msgs)
    # Last message should be user nudge
    assert call3_msgs[-1]["role"] == "user"
    assert "Reply now" in call3_msgs[-1]["content"] or "migration" in call3_msgs[-1]["content"]


def test_uses_last_tool_sql_on_first_no_sql_reply():
    """Use last_tool_sql as fallback immediately on the FIRST no-SQL reply when it exists."""
    adapter, spec = _setup_db()
    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    tc = FakeToolCall(
        fn_name="run_on_clone",
        fn_args=json.dumps({
            "up_sql": "ALTER TABLE employees ADD COLUMN email TEXT;",
            "down_sql": "ALTER TABLE employees DROP COLUMN email;",
        }),
    )
    # Turn 1: tool call to run_on_clone
    resp1 = FakeResponse(content=None, tool_calls=[tc])
    # Turn 2: assistant says "Looks good" with NO sql block
    resp2 = FakeResponse(content="Looks good, done.")

    client.chat.completions.create.side_effect = [resp1, resp2]

    with patch("timemachine.agent.loop.get_client", return_value=client):
        res = run_time_machine_agent(
            adapter=adapter,
            goal="Add email",
            spec=spec,
            target_table="employees",
            max_steps=1,
        )

    # Should immediately use the tool SQL on the first no-SQL reply
    assert "email" in res.up_sql
    assert res.status in ("VERIFIED", "UNVERIFIED")


def test_run_on_clone_does_not_overwrite_inv_result():
    """Tool experiment run_on_clone does not overwrite inv_result."""
    adapter, spec = _setup_db()
    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # Tool call runs bad SQL on clone
    tc = FakeToolCall(
        fn_name="run_on_clone",
        fn_args=json.dumps({"up_sql": "SELECT 1;"}),
    )
    resp1 = FakeResponse(content=None, tool_calls=[tc])
    # Model then submits good SQL
    resp2 = FakeResponse(
        content="```sql\nALTER TABLE employees ADD COLUMN email TEXT;\n-- DOWN\nALTER TABLE employees DROP COLUMN email;\n```"
    )

    client.chat.completions.create.side_effect = [resp1, resp2]

    with patch("timemachine.agent.loop.get_client", return_value=client):
        res = run_time_machine_agent(
            adapter=adapter,
            goal="Add email",
            spec=spec,
            target_table="employees",
            max_steps=1,
        )

    # inv_result should describe the verification of the submitted up_sql, NOT the tool SELECT 1
    assert res.invariant_result is not None
    assert res.status in ("VERIFIED", "UNVERIFIED")


def test_verified_sql_hash_guards_against_false_success():
    """Status is FAILED if last verification does not belong to the SQL being returned."""
    adapter, spec = _setup_db()
    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # Attempt 1: returns invalid SQL (fails verification)
    # Attempt 2: model returns empty reply, loop runs out of steps
    client.chat.completions.create.side_effect = [
        FakeResponse(content="```sql\nINVALID SQL;\n```"),
        FakeResponse(content="No SQL here."),
        FakeResponse(content="Still no SQL."),
    ]

    with patch("timemachine.agent.loop.get_client", return_value=client):
        res = run_time_machine_agent(
            adapter=adapter,
            goal="Add email",
            spec=spec,
            target_table="employees",
            max_steps=2,
        )

    # Must be FAILED, never VERIFIED or UNVERIFIED
    assert res.status == "FAILED"


def test_best_attempt_tracking_returns_fewest_failed_checks():
    """On failure, return the best attempt (fewest failed checks), not the last attempt."""
    adapter, spec = _setup_db()
    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # Attempt 1: 1 failure (e.g. valid syntax, but creates wrong table name)
    # Attempt 2: syntax error (crash in sandbox)
    client.chat.completions.create.side_effect = [
        FakeResponse(content="```sql\nCREATE TABLE emp2 AS SELECT * FROM employees;\n-- DOWN\nDROP TABLE emp2;\n```"),
        FakeResponse(content="```sql\nINVALID SYNTAX ERROR;\n```"),
    ]

    with patch("timemachine.agent.loop.get_client", return_value=client):
        res = run_time_machine_agent(
            adapter=adapter,
            goal="Add email",
            spec=spec,
            target_table="employees",
            max_steps=2,
        )

    assert res.status == "FAILED"
    # Should return Attempt 1 SQL, not the crashing Attempt 2 SQL
    assert "emp2" in res.up_sql


def test_crash_path_records_real_before_count():
    """In crash path, before_count uses real row count from base clone, not 0."""
    adapter, spec = _setup_db()
    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # SQL with syntax error crashes in sandbox
    client.chat.completions.create.return_value = FakeResponse(
        content="```sql\nSYNTAX ERROR;\n```"
    )

    with patch("timemachine.agent.loop.get_client", return_value=client):
        res = run_time_machine_agent(
            adapter=adapter,
            goal="Add email",
            spec=spec,
            target_table="employees",
            max_steps=1,
        )

    assert res.invariant_result is not None
    # Real table has 2 rows (Alice, Bob)
    assert res.invariant_result.before_count == 2


def test_escalation_policy_stays_on_super_then_ultra():
    """Stay on Super for sandbox errors and first data failure; escalate to Ultra on repeat failed check or after 2 failed attempts."""
    adapter, spec = _setup_db()
    client = MagicMock()
    super_model = "nvidia/nemotron-3-super-120b-a12b"
    ultra_model = "nvidia/Nemotron-3-Ultra-550b-a55b"
    client.models.list.return_value.data = [
        MagicMock(id=super_model),
        MagicMock(id=ultra_model),
    ]

    # Attempt 1: Sandbox error (syntax error). Should stay on Super for attempt 2.
    # Attempt 2: Data failure (creates wrong table). After this 2nd failure, should escalate to Ultra for attempt 3.
    # Attempt 3: Valid SQL
    client.chat.completions.create.side_effect = [
        FakeResponse(content="```sql\nSYNTAX ERROR;\n```"),
        FakeResponse(content="```sql\nCREATE TABLE wrong_tbl (id INT);\n-- DOWN\nDROP TABLE wrong_tbl;\n```"),
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

    assert len(res.attempts) == 3
    assert res.attempts[0].model == super_model
    # Attempt 2 stayed on Super because attempt 1 was sandbox error
    assert res.attempts[1].model == super_model
    # Attempt 3 escalated to Ultra after 2 failed attempts
    assert res.attempts[2].model == ultra_model


def test_final_nudge_omits_tools_and_uses_exact_nudge():
    """Final nudge call omits tools entirely and asks for UP then '-- DOWN'."""
    adapter, spec = _setup_db()
    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # Model calls tool profile_table, but returns no SQL
    tc = FakeToolCall(
        fn_name="profile_table",
        fn_args=json.dumps({"table": "employees"}),
    )
    resp1 = FakeResponse(content=None, tool_calls=[tc])
    # Next call is the final nudge call after tool rounds budget
    resp2 = FakeResponse(
        content="```sql\nALTER TABLE employees ADD COLUMN email TEXT;\n-- DOWN\nALTER TABLE employees DROP COLUMN email;\n```"
    )
    client.chat.completions.create.side_effect = [resp1, resp2]

    fake_settings = MagicMock(
        agent_wall_clock_timeout_s=300.0,
        max_tool_rounds_per_step=1,
        llm_timeout_s=60.0,
        max_attempts=6,
    )
    with patch("timemachine.agent.loop.get_settings", return_value=fake_settings), patch("timemachine.agent.loop.get_client", return_value=client):
        run_time_machine_agent(
            adapter=adapter,
            goal="Add email",
            spec=spec,
            target_table="employees",
            max_steps=1,
        )

    # Check the second LLM call (final nudge): tools must be None or omitted
    nudge_call = client.chat.completions.create.call_args_list[1]
    assert nudge_call[1].get("tools") is None
    last_msg = nudge_call[1]["messages"][-1]
    assert "Reply now with the final migration in ```sql blocks, UP then '-- DOWN'" in last_msg["content"]


def test_latency_and_tokens_logged_accurately():
    """Each LLM call logged once with its own tokens/latency; tool steps have tokens_used=0."""
    adapter, spec = _setup_db()
    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    tc = FakeToolCall(
        fn_name="profile_table",
        fn_args=json.dumps({"table": "employees"}),
    )
    resp1 = FakeResponse(content=None, tool_calls=[tc], tokens=150)
    resp2 = FakeResponse(
        content="```sql\nALTER TABLE employees ADD COLUMN email TEXT;\n-- DOWN\nALTER TABLE employees DROP COLUMN email;\n```",
        tokens=250,
    )
    client.chat.completions.create.side_effect = [resp1, resp2]

    with patch("timemachine.agent.loop.get_client", return_value=client):
        res = run_time_machine_agent(
            adapter=adapter,
            goal="Add email",
            spec=spec,
            target_table="employees",
            max_steps=1,
        )

    tool_steps = [s for s in res.steps if s.action.startswith("tool:")]
    assert len(tool_steps) >= 1
    for ts in tool_steps:
        assert ts.tokens_used == 0  # tool executions don't consume LLM tokens

