"""Tests for migration planning, spec generation, normalization, and staleness prevention."""

import json
from unittest.mock import MagicMock, patch

import pytest

from timemachine.adapters.schema import ColumnInfo, DatabaseSchema, TableSchema
from timemachine.verify.spec import ColumnMapping, MigrationSpec


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
        self.usage = MagicMock(total_tokens=50)


def create_sample_schema(table_name: str = "employees") -> DatabaseSchema:
    return DatabaseSchema(
        tables={
            table_name: TableSchema(
                name=table_name,
                columns=[
                    ColumnInfo(name="id", type="INTEGER", pk=True),
                    ColumnInfo(name="full_name", type="TEXT"),
                    ColumnInfo(name="department", type="TEXT"),
                ],
                row_count=10,
            )
        }
    )


def test_generate_spec_list_shaped_mappings_and_unknown_transform():
    """List-shaped column_mappings is normalized to dict, unknown transform becomes 'custom', risky_operations coerced to str."""
    from timemachine.agent.planner import generate_spec

    client = MagicMock()
    # Model returns column_mappings as list, risky_operations as objects, unknown transform
    raw_json = json.dumps({
        "goal": "Normalize employees",
        "affected_tables": ["employees"],
        "column_mappings": [
            {
                "new_column": "first_name",
                "source_column": "full_name",
                "transform": "split_name_first",
            },
            {
                "new_column": "secret_code",
                "source_column": "department",
                "transform": "quantum_hash_unrecognized",
            },
        ],
        "risky_operations": [{"action": "DROP_TABLE", "table": "employees_old"}],
        "extra_unknown_key": "to be dropped",
    })
    client.chat.completions.create.return_value = FakeResponse(content=f"```json\n{raw_json}\n```")

    schema = create_sample_schema("employees")
    res = generate_spec(client, "test-model", schema, "Normalize employees", profiles={})

    assert res.error is None
    assert res.spec is not None
    assert isinstance(res.spec.column_mappings, dict)
    assert "employees" in res.spec.column_mappings
    mappings = res.spec.column_mappings["employees"]
    assert len(mappings) == 2
    assert mappings[0].new_column == "first_name"
    assert mappings[0].transform == "split_name_first"
    # Unrecognized transform coerced to 'custom'
    assert mappings[1].transform == "custom"
    # risky_operations coerced to list of str
    assert all(isinstance(r, str) for r in res.spec.risky_operations)


def test_generate_spec_validation_failure_retries_once():
    """On validation failure, generate_spec retries once with the pydantic error appended."""
    from timemachine.agent.planner import generate_spec

    client = MagicMock()
    # Call 1: invalid JSON syntax or bad schema
    resp1 = FakeResponse(content="```json\n{broken json\n```")
    # Call 2: corrected valid MigrationSpec JSON
    valid_json = json.dumps({
        "goal": "Split name",
        "affected_tables": ["employees"],
        "column_mappings": {
            "employees": [
                {"new_column": "first_name", "source_column": "full_name", "transform": "split_name_first"}
            ]
        },
    })
    resp2 = FakeResponse(content=f"```json\n{valid_json}\n```")
    client.chat.completions.create.side_effect = [resp1, resp2]

    schema = create_sample_schema("employees")
    res = generate_spec(client, "test-model", schema, "Split name", profiles={})

    assert client.chat.completions.create.call_count == 2
    # Verify second call contained error info
    retry_msgs = client.chat.completions.create.call_args_list[1][1]["messages"]
    retry_user_msg = retry_msgs[-1]["content"]
    assert "validation error" in retry_user_msg.lower() or "error" in retry_user_msg.lower()
    assert res.spec is not None
    assert res.spec.goal == "Split name"


def test_generate_spec_two_scenarios_different_specs():
    """Two different scenarios and goals result in different generated specs without copying fixed fallback."""
    from timemachine.agent.planner import generate_spec

    client = MagicMock()

    def fake_create(messages, **kwargs):
        # Echo back a spec containing the goal from prompt
        user_prompt = messages[-1]["content"]
        if "Scenario 1" in user_prompt:
            spec_dict = {
                "goal": "Scenario 1: Dept Normalization",
                "affected_tables": ["departments", "employees"],
                "risky_operations": ["CREATE TABLE departments"],
                "column_mappings": {
                    "employees": [{"new_column": "dept_code", "source_column": "department", "transform": "passthrough"}]
                },
            }
        else:
            spec_dict = {
                "goal": "Scenario 2: Salary Banding",
                "affected_tables": ["employees"],
                "column_mappings": {
                    "employees": [{"new_column": "salary_cents", "source_column": "full_name", "transform": "cast_cents"}]
                },
            }
        return FakeResponse(content=f"```json\n{json.dumps(spec_dict)}\n```")

    client.chat.completions.create.side_effect = fake_create

    schema = create_sample_schema("employees")
    res1 = generate_spec(client, "test-model", schema, "Goal for Scenario 1", profiles={})
    res2 = generate_spec(client, "test-model", schema, "Goal for Scenario 2", profiles={})

    assert res1.spec.goal != res2.spec.goal
    assert res1.spec.affected_tables != res2.spec.affected_tables


def test_generate_spec_failure_returns_error_and_no_spec():
    """When model repeatedly produces unparseable garbage, returns error and no spec."""
    from timemachine.agent.planner import generate_spec

    client = MagicMock()
    client.chat.completions.create.return_value = FakeResponse(content="Sorry, cannot write JSON.")
    schema = create_sample_schema("employees")
    res = generate_spec(client, "test-model", schema, "Invalid goal", profiles={})

    assert res.spec is None
    assert res.error is not None


def test_generate_spec_agent_error_stops_without_retry():
    """An AgentError from call_llm (e.g. 401 auth error) halts immediately with 1 call and preserves error message."""
    from timemachine.agent.planner import generate_spec
    from timemachine.config import AgentError

    client = MagicMock()
    # call_llm will raise AgentError
    client.chat.completions.create.side_effect = Exception("401 Unauthorized: Invalid API key")

    schema = create_sample_schema("employees")
    res = generate_spec(client, "test-model", schema, "Goal", profiles={})

    assert client.chat.completions.create.call_count == 1
    assert res.spec is None
    assert res.source == "error"
    assert "Authentication failed" in (res.error or "")


def test_extract_json_from_text_think_and_raw_decode():
    """extract_json_from_text strips <think> tags and parses first balanced {...} if no fence exists."""
    from timemachine.agent.planner import extract_json_from_text

    text_with_think = "<think>Analyzing table structure...</think>```json\n{\"affected_tables\": [\"employees\"]}\n```"
    data1 = extract_json_from_text(text_with_think)
    assert data1 == {"affected_tables": ["employees"]}

    text_no_fence = "Here is the plan you requested:\n{\"affected_tables\": [\"employees\"], \"goal\": \"split\"}\nLet me know if you need changes."
    data2 = extract_json_from_text(text_no_fence)
    assert data2 == {"affected_tables": ["employees"], "goal": "split"}


def test_validate_spec_target_table_not_in_affected_tables():
    from timemachine.agent.planner import validate_spec_against_schema
    schema = create_sample_schema("employees")
    spec = MigrationSpec(affected_tables=["departments"], goal="test")
    errs = validate_spec_against_schema(spec, schema, "employees")
    assert any("Target table 'employees' is not in affected_tables" in e for e in errs)


def test_validate_spec_nonexistent_affected_table():
    from timemachine.agent.planner import validate_spec_against_schema
    schema = create_sample_schema("employees")
    spec = MigrationSpec(affected_tables=["employees", "ghost_table"], goal="test")
    errs = validate_spec_against_schema(spec, schema, "employees")
    assert any("ghost_table" in e for e in errs)

    # But if declared as created in risky_operations, it is valid
    spec_created = MigrationSpec(
        affected_tables=["employees", "new_table"],
        risky_operations=["CREATE TABLE new_table (id INT)"],
        goal="test",
    )
    errs2 = validate_spec_against_schema(spec_created, schema, "employees")
    assert not any("new_table" in e for e in errs2)


def test_validate_spec_nonexistent_source_column():
    from timemachine.agent.planner import validate_spec_against_schema
    schema = create_sample_schema("employees")
    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="first_name", source_column="ghost_col")
            ]
        },
        goal="test",
    )
    errs = validate_spec_against_schema(spec, schema, "employees")
    assert any("ghost_col" in e for e in errs)


def test_validate_spec_lookup_table_and_keys():
    from timemachine.agent.planner import validate_spec_against_schema
    schema = create_sample_schema("employees")
    # Lookup table does not exist
    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(
                    new_column="dept_name",
                    source_column="department",
                    transform="map_via_lookup",
                    lookup_table="ghost_dept",
                    lookup_key="code",
                    lookup_val="name",
                )
            ]
        },
        goal="test",
    )
    errs = validate_spec_against_schema(spec, schema, "employees")
    assert any("ghost_dept" in e for e in errs)


def test_validate_spec_unchanged_and_constraints_real_columns():
    from timemachine.agent.planner import validate_spec_against_schema
    schema = create_sample_schema("employees")
    spec = MigrationSpec(
        affected_tables=["employees"],
        columns_declared_unchanged={"employees": ["nonexistent_col"]},
        expected_constraints={"employees": ["not_a_col"]},
        goal="test",
    )
    errs = validate_spec_against_schema(spec, schema, "employees")
    assert any("nonexistent_col" in e for e in errs)
    assert any("not_a_col" in e for e in errs)


