"""Tests for Item C: Prompts (transform contract from oracle docstrings, SQLite gotchas, dynamic repair checklist, dead code removal)."""

from timemachine.adapters.schema import ColumnInfo, DatabaseSchema, TableSchema
from timemachine.agent import prompts
from timemachine.verify.invariants import InvariantResult
from timemachine.verify.spec import ColumnMapping, MigrationSpec


def _sample_schema() -> DatabaseSchema:
    return DatabaseSchema(
        dialect="sqlite",
        tables={
            "employees": TableSchema(
                name="employees",
                columns=[
                    ColumnInfo(name="id", type="INTEGER", pk=True),
                    ColumnInfo(name="full_name", type="TEXT"),
                    ColumnInfo(name="salary", type="INTEGER"),
                ],
                row_count=10,
            )
        },
    )


def test_build_plan_prompt_is_deleted():
    """Verify dead build_plan_prompt has been deleted."""
    assert not hasattr(prompts, "build_plan_prompt")


def test_transform_contract_generated_from_oracle_docstrings():
    """Transform contract describes exact oracle check derived from oracle.py docstrings."""
    spec = MigrationSpec(
        goal="Split name and categorize salary",
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="first_name", source_column="full_name", transform="split_name_first"),
                ColumnMapping(new_column="last_name", source_column="full_name", transform="split_name_last", default_value="N/A"),
                ColumnMapping(new_column="band", source_column="salary", transform="salary_band"),
            ]
        },
    )

    contract = prompts.generate_transform_contract(spec)
    assert "Transform Contract" in contract
    assert "first_name" in contract
    assert "split_name_first" in contract
    # Must describe stripping whitespace and splitting
    assert "str.strip()" in contract or "strip" in contract.lower()
    assert "first space" in contract.lower()
    assert "last_name" in contract
    assert "N/A" in contract


def test_sqlite_gotchas_block_present_in_prompt():
    """Prompt contains the SQLite gotchas block with TRIM whitespace, INSTR, NULL concat, PRAGMAs, etc."""
    schema = _sample_schema()
    spec = MigrationSpec(goal="Test", affected_tables=["employees"])

    prompt = prompts.build_migration_prompt(schema, spec)
    assert "SQLite Gotchas" in prompt or "SQLite gotchas" in prompt
    assert "char(9)" in prompt or "char(10)" in prompt or "char(13)" in prompt
    assert "INSTR" in prompt and "0" in prompt
    assert "NULL" in prompt
    assert "REGEXP" in prompt
    assert "AUTOINCREMENT" in prompt or "sqlite_sequence" in prompt
    assert "indexes" in prompt
    assert "PRAGMA" in prompt


def test_prompt_instructs_no_get_schema_and_tool_limit():
    """Prompt tells model: 'The schema is already above; do not call get_schema. You have at most N tool calls; finish with submit.'"""
    schema = _sample_schema()
    spec = MigrationSpec(goal="Test", affected_tables=["employees"])

    prompt = prompts.build_migration_prompt(schema, spec, max_tool_calls=6)
    assert "The schema is already above; do not call get_schema." in prompt
    assert "You have at most 6 tool calls; finish with submit." in prompt

    repair_p = prompts.build_repair_prompt(
        schema, spec, "FAILED SQL",
        result=InvariantResult(status="FAILED", failed_checks=["Syntax error"]),
        max_tool_calls=6,
    )
    assert "The schema is already above; do not call get_schema." in repair_p
    assert "You have at most 6 tool calls; finish with submit." in repair_p


def test_repair_prompt_dynamic_checklist_by_failed_check_types():
    """Task checklist is dynamically constructed from failed check types, not generic hardcoded advice."""
    schema = _sample_schema()
    spec = MigrationSpec(goal="Test", affected_tables=["employees"])

    # Case 1: ROW_LOSS failure -> should give LEFT JOIN advice
    res_row_loss = InvariantResult(
        status="FAILED",
        failed_checks=["ROW_COUNT_MISMATCH: Target lost 5 rows"],
    )
    prompt1 = prompts.build_repair_prompt(schema, spec, "SELECT 1;", result=res_row_loss)
    assert "LEFT JOIN" in prompt1

    # Case 2: LOST_INDEX failure only -> should advise recreating index, NOT LEFT JOIN
    res_index = InvariantResult(
        status="FAILED",
        failed_checks=["LOST_INDEX: index idx_emp_salary was dropped"],
    )
    prompt2 = prompts.build_repair_prompt(schema, spec, "SELECT 1;", result=res_index)
    assert "index" in prompt2.lower()
    assert "LEFT JOIN and COALESCE" not in prompt2

    # Case 3: SANDBOX VIOLATION failure -> should list permitted statements / PRAGMAs
    res_sandbox = InvariantResult(
        status="FAILED",
        failed_checks=["SANDBOX VIOLATION: PRAGMA writable_schema is not permitted."],
    )
    prompt3 = prompts.build_repair_prompt(schema, spec, "SELECT 1;", result=res_sandbox)
    assert "permitted" in prompt3.lower()
    assert "foreign_keys" in prompt3
