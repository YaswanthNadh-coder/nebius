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


def test_parse_sql_scripts_three_blocks_concatenated_as_up():
    from timemachine.agent.loop import parse_sql_scripts
    content = """
```sql
ALTER TABLE employees ADD COLUMN first_name TEXT;
```
```sql
UPDATE employees SET first_name = 'Alice';
```
```sql
ALTER TABLE employees DROP COLUMN old_col;
```
"""
    up, down = parse_sql_scripts(content)
    assert "ADD COLUMN first_name" in up
    assert "UPDATE employees" in up
    assert "DROP COLUMN old_col" in up
    assert down == ""


def test_parse_sql_scripts_up_plus_select_check_block():
    from timemachine.agent.loop import parse_sql_scripts
    content = """
```sql
ALTER TABLE employees ADD COLUMN dept TEXT;
```
```sql
SELECT COUNT(*) FROM employees;
```
"""
    up, down = parse_sql_scripts(content)
    assert "ADD COLUMN dept" in up
    assert down == ""


def test_parse_sql_scripts_down_migration_inside_one_fence():
    from timemachine.agent.loop import parse_sql_scripts
    content = """
```sql
ALTER TABLE employees ADD COLUMN dept TEXT;
-- DOWN MIGRATION
ALTER TABLE employees DROP COLUMN dept;
```
"""
    up, down = parse_sql_scripts(content)
    assert "ADD COLUMN dept" in up
    assert "DROP COLUMN dept" in down


def test_parse_sql_scripts_rollback_prose_before_block():
    from timemachine.agent.loop import parse_sql_scripts
    content = """
Migration:
```sql
ALTER TABLE employees ADD COLUMN dept TEXT;
```
Rollback:
```sql
ALTER TABLE employees DROP COLUMN dept;
```
"""
    up, down = parse_sql_scripts(content)
    assert "ADD COLUMN dept" in up
    assert "DROP COLUMN dept" in down


def test_parse_sql_scripts_no_down_at_all():
    from timemachine.agent.loop import parse_sql_scripts
    content = """
```sql
ALTER TABLE employees ADD COLUMN dept TEXT;
```
"""
    up, down = parse_sql_scripts(content)
    assert "ADD COLUMN dept" in up
    assert down == ""


def test_parse_sql_scripts_reject_down_without_ddl_or_dml():
    from timemachine.agent.loop import parse_sql_scripts
    content = """
```sql
ALTER TABLE employees ADD COLUMN dept TEXT;
-- DOWN
SELECT * FROM employees;
```
"""
    up, down = parse_sql_scripts(content)
    assert "ADD COLUMN dept" in up
    assert down == ""


def test_agent_stops_on_spec_invalid():
    """Agent validates user-supplied spec against schema and stops with SPEC_INVALID if checks fail."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.commit()

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    invalid_spec = MigrationSpec(
        affected_tables=["ghost_table"],  # Target table not in affected tables & table doesn't exist
        goal="Invalid migration",
    )

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Invalid migration",
            spec=invalid_spec,
            target_table="employees",
            api_key_override="test_key_123456",
        )

    assert result.status == "FAILED"
    assert result.stop_reason == "SPEC_INVALID"
    assert "ghost_table" in result.notes or "Target table" in result.notes


def test_agent_stops_on_no_progress_identical_sql():
    """When a model returns identical failing SQL, the run stops with NO_PROGRESS after at most 3 LLM calls."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.execute("INSERT INTO employees VALUES (1, 'Alice');")
    conn.commit()

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # Always returns the exact same invalid SQL statement that fails
    identical_sql_reply = "```sql\nALTER TABLE employees ADD COLUMN first_name TEXT NOT NULL;\n```"
    client.chat.completions.create.return_value = FakeResponse(content=identical_sql_reply)

    spec = MigrationSpec(
        affected_tables=["employees"],
        goal="Split name",
    )

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Split name",
            spec=spec,
            target_table="employees",
            api_key_override="test_key_123456",
            max_steps=8,
        )

    assert result.status == "FAILED"
    assert result.stop_reason == "NO_PROGRESS"
    assert client.chat.completions.create.call_count <= 3


def test_tool_set_inv_result_tracks_last_tool_sql_and_preserves_draft_label():
    """Tool-set inv_result preserves last_tool_sql for the next prompt and keeps draft_migration label."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.commit()

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    # Step 1: Model calls run_on_clone with tool SQL, but outputs no SQL in the step response
    tc = FakeToolCall(
        fn_name="run_on_clone",
        fn_args=json.dumps({"up_sql": "ALTER TABLE employees ADD COLUMN tool_test_col TEXT;"}),
        call_id="call_clone_1",
    )
    # Turn 1: tool call
    resp1 = FakeResponse(content=None, tool_calls=[tc])
    # Turn 2 in Step 1: assistant says "Done checking" without any SQL block
    resp2 = FakeResponse(content="Done checking clone.")

    # Step 2: Model returns SQL
    resp3 = FakeResponse(
        content="```sql\nALTER TABLE employees ADD COLUMN first_name TEXT DEFAULT 'ok';\n```"
    )

    client.chat.completions.create.side_effect = [resp1, resp2, resp3]

    spec = MigrationSpec(affected_tables=["employees"], goal="Split name")

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Split name",
            spec=spec,
            target_table="employees",
            api_key_override="test_key_123456",
            max_steps=4,
        )

    # Check Step 2 prompt:
    # 3rd create call was for Step 2. Its prompt messages should contain 'tool_test_col'
    step2_msgs = client.chat.completions.create.call_args_list[2][1]["messages"]
    step2_prompt = step2_msgs[0]["content"]
    assert "tool_test_col" in step2_prompt, f"last_tool_sql not included in repair prompt: {step2_prompt}"

    # Verify action labels: step that produces the first migration SQL is labeled draft_migration
    actions = [s.action for s in result.steps]
    assert "draft_migration" in actions, f"Expected draft_migration in actions: {actions}"
    assert "repair_migration" not in actions, f"repair_migration should not precede draft_migration: {actions}"


def test_agent_repair_multi_tool_investigation_allows_completion():
    """Agent undergoing deep repair with >4 tool calls (get_schema, profile, find, run_on_clone, submit) completes successfully without being cut off."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.execute("INSERT INTO employees VALUES (1, 'Alice');")
    conn.commit()

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    spec = MigrationSpec(affected_tables=["employees"], goal="Add notes")

    # Call 1 (Super): drafts bad SQL (fails)
    resp1 = FakeResponse(content="```sql\nINVALID SQL;\n```")

    # Step 2 (Ultra repair): calls 4 investigation tools, then submit on round 5!
    tc_schema = FakeToolCall("get_schema", "{}", "tc1")
    resp_schema = FakeResponse(content=None, tool_calls=[tc_schema])

    tc_profile = FakeToolCall("profile_table", json.dumps({"table_name": "employees"}), "tc2")
    resp_profile = FakeResponse(content=None, tool_calls=[tc_profile])

    tc_find = FakeToolCall("find_violating_rows", json.dumps({"table_name": "employees", "predicate": "1=1"}), "tc3")
    resp_find = FakeResponse(content=None, tool_calls=[tc_find])

    tc_clone = FakeToolCall("run_on_clone", json.dumps({"up_sql": "ALTER TABLE employees ADD COLUMN notes TEXT DEFAULT '';"}), "tc4")
    resp_clone = FakeResponse(content=None, tool_calls=[tc_clone])

    # Round 5: submit verified SQL
    tc_submit = FakeToolCall("submit", json.dumps({
        "up_sql": "ALTER TABLE employees ADD COLUMN notes TEXT DEFAULT '';",
        "down_sql": "",
    }), "tc5")
    resp_submit = FakeResponse(content=None, tool_calls=[tc_submit])

    client.chat.completions.create.side_effect = [
        resp1,
        resp_schema,
        resp_profile,
        resp_find,
        resp_clone,
        resp_submit,
    ]

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Add notes",
            spec=spec,
            target_table="employees",
            api_key_override="test_key_123456",
            max_steps=2,
        )

    assert result.status in ("VERIFIED", "UNVERIFIED")
    assert result.stop_reason != "MODEL_RETURNED_NO_SQL"
    assert "ADD COLUMN notes" in result.up_sql
    actions = [s.action for s in result.steps]
    assert "tool:submit" in actions
    assert "no_sql_reply" not in actions


def test_agent_tool_budget_exhaustion_triggers_final_completion():
    """When tool calling reaches max_tool_rounds with dangling tool output, a final completion call allows final SQL."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.commit()

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    spec = MigrationSpec(affected_tables=["employees"], goal="Add notes")

    # Step 1: Model calls tools up to budget, and then final turn produces SQL in response to reminder
    tc1 = FakeToolCall("get_schema", "{}", "call_1")
    resp1 = FakeResponse(content=None, tool_calls=[tc1])

    tc2 = FakeToolCall("profile_table", json.dumps({"table_name": "employees"}), "call_2")
    resp2 = FakeResponse(content=None, tool_calls=[tc2])

    # Final completion response (no tool calls, text with SQL)
    resp_final = FakeResponse(content="```sql\nALTER TABLE employees ADD COLUMN notes TEXT DEFAULT '';\n```")

    client.chat.completions.create.side_effect = [resp1, resp2, resp_final]

    mock_settings = MagicMock(
        max_tool_rounds_per_step=2,
        agent_wall_clock_timeout_s=180.0,
        llm_timeout_s=90.0,
        max_llm_tokens=8192,
        default_fast_model="nvidia/nemotron-3-super-120b-a12b",
        default_ultra_model="nvidia/Nemotron-3-Ultra-550b-a55b",
    )

    with patch("timemachine.agent.loop.get_client", return_value=client), \
         patch("timemachine.agent.loop.get_settings", return_value=mock_settings):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Add notes",
            spec=spec,
            target_table="employees",
            api_key_override="test_key_123456",
            max_steps=1,
        )

    assert result.status in ("VERIFIED", "UNVERIFIED")
    assert "ADD COLUMN notes" in result.up_sql
    actions = [s.action for s in result.steps]
    assert "draft_migration" in actions
    assert "no_sql_reply" not in actions


def test_agent_falls_back_to_last_tool_sql_before_no_sql_abort():
    """When model tests SQL via run_on_clone but provides no fenced SQL in consecutive turns, loop falls back to last_tool_sql rather than failing with MODEL_RETURNED_NO_SQL."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.commit()

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    spec = MigrationSpec(affected_tables=["employees"], goal="Add notes")

    # Step 1: Model calls run_on_clone with valid SQL, then text has no SQL
    tc1 = FakeToolCall("run_on_clone", json.dumps({"up_sql": "ALTER TABLE employees ADD COLUMN notes TEXT DEFAULT '';"}), "tc1")
    resp1 = FakeResponse(content=None, tool_calls=[tc1])
    resp2 = FakeResponse(content="Done checking.")

    # Step 2: Model text has no SQL again
    resp3 = FakeResponse(content="Still checking without SQL.")

    client.chat.completions.create.side_effect = [resp1, resp2, resp3]

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Add notes",
            spec=spec,
            target_table="employees",
            api_key_override="test_key_123456",
            max_steps=3,
        )

    # Should not fail with MODEL_RETURNED_NO_SQL because last_tool_sql was used
    assert result.stop_reason != "MODEL_RETURNED_NO_SQL"
    assert "ADD COLUMN notes" in result.up_sql
    assert result.status in ("VERIFIED", "UNVERIFIED")


def test_agent_preserves_conversation_messages_on_no_sql_retry():
    """When model returns no SQL after tool execution, retry preserves prior tool messages in context."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT);")
    conn.commit()

    client = MagicMock()
    client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b"),
    ]

    spec = MigrationSpec(affected_tables=["employees"], goal="Add notes")

    tc1 = FakeToolCall("get_schema", "{}", "tc1")
    resp1 = FakeResponse(content=None, tool_calls=[tc1])
    resp2 = FakeResponse(content="I see the schema.")

    # Retry turn
    resp3 = FakeResponse(
        content="```sql\n-- UP MIGRATION\nALTER TABLE employees ADD COLUMN notes TEXT DEFAULT '';\n```"
    )

    client.chat.completions.create.side_effect = [resp1, resp2, resp3]

    recorded_messages = []

    def mock_create(*args, **kwargs):
        if "messages" in kwargs:
            recorded_messages.append(list(kwargs["messages"]))
        return client.chat.completions.create.side_effect.pop(0)

    client.chat.completions.create = MagicMock(side_effect=[resp1, resp2, resp3])

    with patch("timemachine.agent.loop.get_client", return_value=client):
        result = run_time_machine_agent(
            adapter=adapter,
            goal="Add notes",
            spec=spec,
            target_table="employees",
            api_key_override="test_key_123456",
            max_steps=2,
        )

    assert result.status in ("VERIFIED", "UNVERIFIED")
    # Verify call 3 received multi-turn history with tool response
    third_call_messages = client.chat.completions.create.call_args_list[2].kwargs["messages"]
    roles = [m["role"] for m in third_call_messages]
    assert "tool" in roles
    assert roles[-1] == "user"
    assert "no executable migration SQL" in third_call_messages[-1]["content"]


def test_run_on_clone_handles_read_queries():
    """When model passes a read-only query to run_on_clone, it returns query_result with sample rows instead of failing invariants."""
    from timemachine.agent.tools import ToolExecutor

    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.open_readonly()
    conn.execute("CREATE TABLE departments (dept_code TEXT PRIMARY KEY, dept_name TEXT);")
    conn.execute("INSERT INTO departments VALUES ('ENG', 'Engineering'), ('MKT', 'Marketing');")
    conn.commit()

    executor = ToolExecutor(adapter, target_table="employees")
    res = executor.execute("run_on_clone", json.dumps({"up_sql": "SELECT dept_code, dept_name FROM departments ORDER BY dept_code;"}))

    assert res.get("status") == "query_result"
    assert "dept_code" in res.get("columns", [])
    assert res.get("count") == 2
    assert len(res.get("sample_rows", [])) == 2


def test_call_llm_reports_actual_timeout_in_error():
    """When call_llm encounters APITimeoutError, the error message reflects the specific timeout passed."""
    from openai import APITimeoutError

    from timemachine.agent.llm import call_llm

    client = MagicMock()
    client.chat.completions.create.side_effect = APITimeoutError("Request timed out")

    with pytest.raises(AgentError) as exc_info:
        call_llm(client, "test-model", "test prompt", timeout=12.5)

    assert "timed out (12s)" in str(exc_info.value)












