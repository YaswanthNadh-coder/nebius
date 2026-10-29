"""Tests for Item C: Spec Quality Gate, null_policy auto-fix/validation, orphan_policy, and satisfiability self-checks."""

from timemachine.adapters.schema import ColumnInfo, DatabaseSchema, TableSchema
from timemachine.agent.planner import auto_fix_spec_null_policies, validate_spec_against_schema
from timemachine.agent.prompts import generate_transform_contract
from timemachine.scenarios import get_builtin_scenario_reference
from timemachine.traps.profiler import ColumnProfile
from timemachine.ui.app_view import create_builtin_scenario_adapter
from timemachine.verify.invariants import evaluate_migration_invariants
from timemachine.verify.spec import ColumnMapping, MigrationSpec


def test_spec_autofix_nullable_passthrough():
    """A passthrough mapping on a nullable column must get null_policy='nullable'."""
    schema = DatabaseSchema(
        tables={
            "employees": TableSchema(
                name="employees",
                columns=[
                    ColumnInfo(name="id", type="INTEGER", notnull=True, pk=1),
                    ColumnInfo(name="salary", type="INTEGER", notnull=False, pk=0),  # nullable
                ],
            )
        }
    )

    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="salary", source_column="salary", transform="passthrough", null_policy="not_null"),
            ]
        },
    )

    auto_fix_spec_null_policies(spec, schema)
    mapping = spec.column_mappings["employees"][0]
    assert mapping.null_policy == "nullable", "Passthrough nullable column should have been auto-fixed to 'nullable'"


def test_spec_rejects_not_null_with_profile_nulls():
    """Raise validation error when mapping is not_null, profile has NULLs, and no default exists."""
    schema = DatabaseSchema(
        tables={
            "employees": TableSchema(
                name="employees",
                columns=[
                    ColumnInfo(name="id", type="INTEGER", notnull=True, pk=1),
                    ColumnInfo(name="dept", type="TEXT", notnull=False, pk=0),
                ],
            )
        }
    )

    profiles = {
        "employees": {
            "dept": ColumnProfile(
                column_name="dept",
                total_count=10,
                null_count=3,
                distinct_count=4,
            )
        }
    }

    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(new_column="dept", source_column="dept", transform="passthrough", null_policy="not_null"),
            ]
        },
    )

    errors = validate_spec_against_schema(spec, schema, target_table="employees", profiles=profiles)
    assert any("NULL" in err and "dept" in err for err in errors), f"Expected NULL profile validation error, got {errors}"


def test_orphan_policy_in_column_mapping_and_contract():
    """ColumnMapping supports orphan_policy and reflects it in the Transform Contract."""
    spec = MigrationSpec(
        affected_tables=["employees"],
        column_mappings={
            "employees": [
                ColumnMapping(
                    new_column="dept_code",
                    source_column="dept",
                    transform="map_via_lookup",
                    lookup_table="departments",
                    lookup_key="code",
                    lookup_val="name",
                    orphan_policy="null",
                ),
            ]
        },
    )

    contract = generate_transform_contract(spec)
    assert "orphan_policy" in contract
    assert "null" in contract


def test_builtin_scenarios_reference_migrations_pass_spec():
    """Satisfiability: correct reference migrations pass invariants for every built-in scenario."""
    # Scenario 1
    adapter1 = create_builtin_scenario_adapter(1)
    ref_up_1, ref_down_1, spec1 = get_builtin_scenario_reference(1)
    base_1 = adapter1.clone()
    test_1 = adapter1.clone()
    adapter1.run_in_sandbox(test_1, ref_up_1, fk_enabled=True)
    inv1 = evaluate_migration_invariants(base_1, test_1, target_table="employees", spec=spec1)
    assert inv1.passed is True, f"Scenario 1 reference migration failed: {inv1.failed_checks}"

    # Scenario 2
    adapter2 = create_builtin_scenario_adapter(2)
    ref_up_2, ref_down_2, spec2 = get_builtin_scenario_reference(2)
    base_2 = adapter2.clone()
    test_2 = adapter2.clone()
    adapter2.run_in_sandbox(test_2, ref_up_2, fk_enabled=True)
    inv2 = evaluate_migration_invariants(base_2, test_2, target_table="employees", spec=spec2)
    assert inv2.passed is True, f"Scenario 2 reference migration failed: {inv2.failed_checks}"
