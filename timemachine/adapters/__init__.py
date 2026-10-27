"""Database adapters package."""

from timemachine.adapters.base import DbAdapter
from timemachine.adapters.dialect import classify_sql, detect_dangerous_operations
from timemachine.adapters.postgres import PostgresAdapter
from timemachine.adapters.schema import (
    ColumnInfo,
    DatabaseSchema,
    ForeignKeyInfo,
    IndexInfo,
    TableSchema,
    TriggerInfo,
    ViewInfo,
)
from timemachine.adapters.sqlite import (
    SQLiteAdapter,
    install_sandbox,
    make_authorizer,
    make_progress_handler,
)

__all__ = [
    "DbAdapter",
    "SQLiteAdapter",
    "PostgresAdapter",
    "DatabaseSchema",
    "TableSchema",
    "ColumnInfo",
    "ForeignKeyInfo",
    "IndexInfo",
    "ViewInfo",
    "TriggerInfo",
    "install_sandbox",
    "make_authorizer",
    "make_progress_handler",
    "classify_sql",
    "detect_dangerous_operations",
]
