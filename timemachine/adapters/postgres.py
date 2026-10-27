"""Postgres adapter placeholder (feature-flagged stretch goal)."""

import sqlite3

from timemachine.adapters.base import DbAdapter
from timemachine.adapters.schema import DatabaseSchema
from timemachine.config import AgentError


class PostgresAdapter(DbAdapter):
    """Postgres adapter for transactional schema migrations."""

    def __init__(self, connection_url: str | None = None, enabled: bool = False):
        self.connection_url = connection_url
        self.enabled = enabled

    def open_readonly(self) -> None:
        if not self.enabled:
            raise AgentError(
                "Postgres adapter is currently feature-flagged. "
                "Enable with TIMEMACHINE_ENABLE_POSTGRES=1."
            )
        raise NotImplementedError("Postgres connectivity requires Docker testcontainers.")

    def introspect(self) -> DatabaseSchema:
        raise NotImplementedError

    def profile(self, table: str) -> dict:
        raise NotImplementedError

    def clone(self) -> sqlite3.Connection:
        raise NotImplementedError

    def run_in_sandbox(
        self,
        clone_conn: sqlite3.Connection,
        sql: str,
        fk_enabled: bool = True,
    ) -> tuple[bool, str | None, float]:
        raise NotImplementedError

    def source_bytes_hash(self) -> str:
        return "pg_unsupported"

    def close(self) -> None:
        pass
