"""Migration specification schema and validation."""

from typing import Any, Literal

from pydantic import BaseModel, Field

TransformType = Literal[
    "passthrough",
    "split_name_first",
    "split_name_last",
    "strip_honorific",
    "cast_int",
    "cast_cents",
    "default_if_null",
    "map_via_lookup",
    "normalize_date",
    "salary_band",
    "trim",
    "lower",
    "upper",
    "custom",
]


class ColumnMapping(BaseModel):
    """Mapping rule from source column(s) to a target column."""

    new_column: str
    source_column: str | None = None
    transform: TransformType = "passthrough"
    null_policy: str = "nullable"   # not_null, nullable
    default_value: Any | None = None
    orphan_policy: str = "null"     # "null", "default:<value>", "reject"
    lookup_table: str | None = None
    lookup_key: str | None = None
    lookup_val: str | None = None


class MigrationSpec(BaseModel):
    """Structured plan specifying the intended schema migration."""

    goal: str = ""
    affected_tables: list[str] = Field(default_factory=list)
    column_mappings: dict[str, list[ColumnMapping]] = Field(default_factory=dict)
    row_policy: str = "preserve_all"  # preserve_all or deletions_allowed
    deletions_justification: str | None = None
    columns_declared_unchanged: dict[str, list[str]] = Field(default_factory=dict)
    expected_constraints: dict[str, list[str]] = Field(default_factory=dict)
    risky_operations: list[str] = Field(default_factory=list)
    user_approved: bool = False

    def get_table_mappings(self, table: str) -> list[ColumnMapping]:
        """Get column mappings for a specific table."""
        return self.column_mappings.get(table, [])

    def get_required_columns(self, table: str) -> list[str]:
        """List of target columns expected for the table."""
        mappings = self.get_table_mappings(table)
        if mappings:
            return [m.new_column for m in mappings]
        return []

    def get_non_null_columns(self, table: str) -> list[str]:
        """List of columns that must be NOT NULL according to the spec."""
        mappings = self.get_table_mappings(table)
        unchanged = {c.lower() for c in self.columns_declared_unchanged.get(table, [])}
        res = []
        for m in mappings:
            if m.null_policy == "not_null":
                # An untouched passthrough column cannot be forced NOT NULL
                if m.new_column.lower() in unchanged and (m.transform or "passthrough").lower() in ("passthrough", "identity"):
                    continue
                res.append(m.new_column)
        return res
