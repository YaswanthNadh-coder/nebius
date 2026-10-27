"""Data structures representing database schema metadata."""

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    """Metadata for a single column."""

    name: str
    type: str = "TEXT"
    notnull: bool = False
    default: str | None = None
    pk: int = 0  # 0 if not pk, 1+ for composite pk ordering


class ForeignKeyInfo(BaseModel):
    """Foreign key constraint definition."""

    from_col: str
    to_table: str
    to_col: str
    on_update: str = "NO ACTION"
    on_delete: str = "NO ACTION"


class IndexInfo(BaseModel):
    """Index definition."""

    name: str
    table: str
    unique: bool = False
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)


class ViewInfo(BaseModel):
    """View definition."""

    name: str
    sql: str


class TriggerInfo(BaseModel):
    """Trigger definition."""

    name: str
    table: str
    sql: str


class TableSchema(BaseModel):
    """Schema definition for a single table."""

    name: str
    columns: list[ColumnInfo] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyInfo] = Field(default_factory=list)
    indexes: list[IndexInfo] = Field(default_factory=list)
    row_count: int = 0

    @property
    def pk_columns(self) -> list[str]:
        """List of column names forming the primary key."""
        pks = sorted([c for c in self.columns if c.pk > 0], key=lambda c: c.pk)
        return [c.name for c in pks]

    @property
    def non_null_columns(self) -> list[str]:
        """List of column names with NOT NULL constraint."""
        return [c.name for c in self.columns if c.notnull]

    def get_column(self, name: str) -> ColumnInfo | None:
        """Find column by name."""
        for c in self.columns:
            if c.name.lower() == name.lower():
                return c
        return None


class DatabaseSchema(BaseModel):
    """Full database schema metadata."""

    dialect: str = "sqlite"
    tables: dict[str, TableSchema] = Field(default_factory=dict)
    views: dict[str, ViewInfo] = Field(default_factory=dict)
    triggers: dict[str, TriggerInfo] = Field(default_factory=dict)
    initial_foreign_keys: bool = True

    def get_table(self, name: str) -> TableSchema | None:
        """Lookup table case-insensitively."""
        for tname, tbl in self.tables.items():
            if tname.lower() == name.lower():
                return tbl
        return None
