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
