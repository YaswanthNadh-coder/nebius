"""Agent tests with scripted fake LLM client (offline, no network)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.agent.llm import call_llm
from timemachine.agent.loop import run_time_machine_agent
from timemachine.config import AgentError
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


class FakeToolCall:
    def __init__(self, fn_name, fn_args, call_id="call_123"):
        self.id = call_id
        self.type = "function"
        self.function = MagicMock(name=fn_name, arguments=fn_args)
        self.function.name = fn_name
        self.function.arguments = fn_args


def test_call_llm_finish_reason_length():
    client = MagicMock()
    client.chat.completions.create.return_value = FakeResponse(
        content="Partial SQL", finish_reason="length"
    )
    with pytest.raises(AgentError) as exc:
        call_llm(client, "test-model", "test prompt")
    assert "truncated" in str(exc.value)


def test_call_llm_api_error_translation_401():
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("401 Unauthorized: Invalid API key")
    with pytest.raises(AgentError) as exc:
        call_llm(client, "test-model", "test prompt")
    assert "Authentication failed" in str(exc.value)


def test_call_llm_api_error_translation_429():
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("429 Too Many Requests: Rate limit")
    with pytest.raises(AgentError) as exc:
        call_llm(client, "test-model", "test prompt")
    assert "Rate limit" in str(exc.value)


def test_call_llm_api_error_translation_timeout():
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("Request timed out after 90 seconds")
    with pytest.raises(AgentError) as exc:
        call_llm(client, "test-model", "test prompt")
    assert "timed out" in str(exc.value)


def test_agent_step_budget_exhaustion():
    """Agent terminates cleanly when max_steps budget is reached."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.execute("INSERT INTO employees VALUES (1, 'Alice');")
    conn.commit()

    spec = MigrationSpec(
        goal="Add email column",
        affected_tables=["employees"],
    )

    # Mock client that keeps returning invalid SQL causing invariants to fail
    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]
    # Always returns SQL with syntax error
    client.chat.completions.create.return_value = FakeResponse(
        content="```sql\nINVALID SQL STATEMENTS SYNTAX ERROR;\n```"
    )

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Add email column",
            spec=spec,
            target_table="employees",
            max_steps=2,
            api_key_override="test_key_123456",
        )

    assert result.status == "FAILED"
    assert "STEP_BUDGET" in result.stop_reason or len(result.steps) <= 3


def test_agent_model_escalation_on_failure():
    """Agent drafts with Super, then escalates to Ultra upon verification failure."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.execute("INSERT INTO employees VALUES (1, 'Alice');")
    conn.commit()

    spec = MigrationSpec(
        goal="Add notes column",
        affected_tables=["employees"],
    )

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # Call 1 (Super): returns broken SQL
    # Call 2 (Ultra): returns corrected valid SQL
    resp1 = FakeResponse(content="```sql\nINVALID SYNTAX;\n```")
    resp2 = FakeResponse(content="```sql\nALTER TABLE employees ADD COLUMN notes TEXT DEFAULT '';\n```")
    client.chat.completions.create.side_effect = [resp1, resp2]

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Add notes column",
            spec=spec,
            target_table="employees",
            max_steps=3,
            api_key_override="test_key_123456",
        )

    # Check that model escalated to Ultra on step 2
    assert len(result.steps) >= 2
    models_called = [s.model for s in result.steps]
    assert "nvidia/nemotron-3-super-120b-a12b" in models_called[0]
    assert "nvidia/Nemotron-3-Ultra-550b-a55b" in models_called[1]
    assert result.status in ("VERIFIED", "UNVERIFIED")


def test_no_repair_call_after_successful_verification():
    """When attempt 1 passes verification, NO repair LLM call is made."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.execute("INSERT INTO employees VALUES (1, 'Alice');")
    conn.commit()

    spec = MigrationSpec(
        goal="Add nickname column",
        affected_tables=["employees"],
    )

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # First attempt generates valid SQL
    client.chat.completions.create.return_value = FakeResponse(
        content="```sql\nALTER TABLE employees ADD COLUMN nickname TEXT DEFAULT '';\n```"
    )

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Add nickname column",
            spec=spec,
            target_table="employees",
            max_steps=5,
            api_key_override="test_key_123456",
        )

    assert result.status in ("VERIFIED", "UNVERIFIED")
    # Verify exactly 1 draft call was made, zero repair calls
    assert client.chat.completions.create.call_count == 1


# --- Bug 1 Failing Tests ---

def test_invariant_result_fail_closed_defaults():
    """InvariantResult defaults must be fail-closed (status='FAILED', passed=False)."""
    from timemachine.verify.invariants import InvariantResult
    inv = InvariantResult()
    assert inv.status == "FAILED"
    assert inv.passed is False


def test_build_repair_prompt_none_result():
    """build_repair_prompt must accept result=None and ask for migration."""
    from timemachine.adapters.schema import DatabaseSchema
    from timemachine.agent.prompts import build_repair_prompt
    schema = DatabaseSchema()
    spec = MigrationSpec(affected_tables=["employees"], goal="test")
    prompt = build_repair_prompt(schema, spec, "I am thinking about it...", None)
    assert "your previous reply contained no executable SQL" in prompt
    assert "reply was: I am thinking about it..." in prompt


def test_agent_empty_first_reply_no_crash():
    """When step 1 produces no SQL, agent does not crash with 'NoneType' has no attribute 'failed_checks'."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.execute("INSERT INTO employees VALUES (1, 'Alice');")
    conn.commit()

    spec = MigrationSpec(goal="Add email", affected_tables=["employees"])
    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # Step 1: empty reply. Step 2: valid SQL
    resp1 = FakeResponse(content="Sure, I can help you with that without any SQL.")
    resp2 = FakeResponse(content="```sql\nALTER TABLE employees ADD COLUMN email TEXT DEFAULT '';\n```")
    client.chat.completions.create.side_effect = [resp1, resp2]

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Add email",
            spec=spec,
            target_table="employees",
            max_steps=4,
            api_key_override="test_key_123456",
        )

    assert result.status in ("VERIFIED", "UNVERIFIED")
    # Check that raw replies are recorded in step details
    raw_replies = [s.details.get("raw_reply") for s in result.steps if "raw_reply" in s.details]
    assert len(raw_replies) >= 1


def test_agent_consecutive_no_sql_stops_with_reason():
    """Allow at most 2 consecutive no-SQL replies, then stop with MODEL_RETURNED_NO_SQL."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.commit()

    spec = MigrationSpec(goal="Add email", affected_tables=["employees"])
    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # Model keeps producing conversational text without any SQL
    client.chat.completions.create.return_value = FakeResponse(content="Hello! Here is advice on SQL.")

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Add email",
            spec=spec,
            target_table="employees",
            max_steps=5,
            api_key_override="test_key_123456",
        )

    assert result.status == "FAILED"
    assert result.stop_reason == "MODEL_RETURNED_NO_SQL"
    assert len(result.steps) <= 2


def test_tool_executor_safe_limit_and_try_except():
    """ToolExecutor handles bad arguments safely and returns error dict on failures."""
    from timemachine.agent.tools import ToolExecutor
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.commit()

    executor = ToolExecutor(adapter, target_table="employees")
    res = executor.execute("find_violating_rows", json.dumps({"limit": "five", "predicate": "1=1"}))
    assert "error" in res or "count" in res

    # Check run_on_clone error return sets status FAILED explicitly
    res_clone = executor.execute("run_on_clone", json.dumps({"up_sql": "BAD SQL SYNTAX ERROR;"}))
    assert res_clone.get("status") == "FAILED"
    assert res_clone.get("passed") is False


# --- Bug 2 & Bug 5 Failing Tests ---

class FakeUsage:
    def __init__(self, total_tokens=42):
        self.total_tokens = total_tokens
        self.prompt_tokens = 20
        self.completion_tokens = 22


def test_call_llm_accepts_messages_and_returns_message_and_usage():
    """call_llm must accept messages and return raw assistant message plus usage."""
    client = MagicMock()
    fake_resp = FakeResponse(content="test reply")
    fake_resp.usage = FakeUsage(total_tokens=123)
    client.chat.completions.create.return_value = fake_resp

    messages = [{"role": "user", "content": "hello"}]
    msg, usage = call_llm(client, "test-model", messages=messages)

    assert msg.content == "test reply"
    assert usage.total_tokens == 123
    # Check that client was called with messages
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["messages"] == messages


def test_call_llm_error_mapping_no_false_positives():
    """call_llm does not falsely match 'generate' as rate limit or '4010' as 401."""
    client = MagicMock()
    # Message containing "generate" (has "rate") - should NOT be treated as RateLimit
    client.chat.completions.create.side_effect = Exception("Failed to generate response accurately")
    with pytest.raises(AgentError) as exc:
        call_llm(client, "test-model", "test prompt")
    assert "Rate limit" not in str(exc.value)

    # Message containing "4010" - should NOT be treated as 401
    client.chat.completions.create.side_effect = Exception("Error code 4010: Invalid port number")
    with pytest.raises(AgentError) as exc2:
        call_llm(client, "test-model", "test prompt")
    assert "Authentication failed" not in str(exc2.value)


def test_agent_multi_turn_tool_calling():
    """Tool calls get executed, results are passed back to the model, and total_tokens is tracked."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.commit()

    spec = MigrationSpec(goal="Add email column", affected_tables=["employees"])
    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # Turn 1: Model calls get_schema
    tc = FakeToolCall(fn_name="get_schema", fn_args="{}", call_id="call_abc123")
    resp_tool = FakeResponse(content=None, tool_calls=[tc])
    resp_tool.usage = FakeUsage(total_tokens=50)

    # Turn 2: Model receives tool answer and outputs SQL
    resp_sql = FakeResponse(content="```sql\nALTER TABLE employees ADD COLUMN email TEXT DEFAULT '';\n```")
    resp_sql.usage = FakeUsage(total_tokens=75)

    client.chat.completions.create.side_effect = [resp_tool, resp_sql]

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Add email column",
            spec=spec,
            target_table="employees",
            max_steps=2,
            api_key_override="test_key_123456",
        )

    # Verify model was called twice in multi-turn loop
    assert client.chat.completions.create.call_count == 2
    # Verify second call received the tool response in its messages
    second_call_msgs = client.chat.completions.create.call_args_list[1][1]["messages"]
    tool_msg = next((m for m in second_call_msgs if m.get("role") == "tool"), None)
    assert tool_msg is not None, "Tool response message was not sent back to model"
    assert tool_msg["tool_call_id"] == "call_abc123"
    assert "employees" in tool_msg["content"]

    # Verify total_tokens is accumulated
    assert result.total_tokens >= 125


def test_agent_stops_on_plan_failed():
    """When spec is None and planning fails, agent stops with stop_reason PLAN_FAILED instead of silent fallback."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.commit()

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]
    # Planner returns invalid output
    client.chat.completions.create.return_value = FakeResponse(content="not valid json")

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Rebuild employees table",
            spec=None,
            target_table="employees",
            api_key_override="test_key_123456",
        )

    assert result.status == "FAILED"
    assert result.stop_reason == "PLAN_FAILED"
    assert result.spec is None



