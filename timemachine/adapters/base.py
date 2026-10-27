"""Base interface for database adapters."""

import sqlite3
from abc import ABC, abstractmethod

from timemachine.adapters.schema import DatabaseSchema


class DbAdapter(ABC):
    """Abstract interface for database connection and isolation."""

    @abstractmethod
    def open_readonly(self) -> None:
        """Open the source database in strict read-only mode."""

    @abstractmethod
    def introspect(self) -> DatabaseSchema:
        """Introspect tables, columns, indexes, FKs, views, and triggers."""

    @abstractmethod
    def profile(self, table: str) -> dict:
        """Profile table statistics (null rates, distinct counts, etc.)."""

    @abstractmethod
    def clone(self) -> sqlite3.Connection:
        """Create a disposable in-RAM clone of the database."""

    @abstractmethod
    def run_in_sandbox(
        self,
        clone_conn: sqlite3.Connection,
        sql: str,
        fk_enabled: bool = True,
    ) -> tuple[bool, str | None, float]:
        """Execute SQL inside a sandboxed clone connection."""

    @abstractmethod
    def source_bytes_hash(self) -> str:
        """Return cryptographic hash of source database state to verify zero mutations."""

    @abstractmethod
    def close(self) -> None:
        """Close connections."""
