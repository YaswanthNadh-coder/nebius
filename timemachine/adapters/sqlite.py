"""SQLite database adapter with strict read-only source and RAM cloning."""

import hashlib
import sqlite3
import time
from pathlib import Path

from timemachine.adapters.base import DbAdapter
from timemachine.adapters.schema import (
    ColumnInfo,
    DatabaseSchema,
    ForeignKeyInfo,
    IndexInfo,
    TableSchema,
    TriggerInfo,
    ViewInfo,
)
from timemachine.config import AgentError, get_settings
from timemachine.logging import get_logger
from timemachine.verify.sandbox import (
    ALLOWED_PRAGMAS,
    SandboxAuthorizer,
    calc_sandbox_timeout,
    make_sandbox_authorizer,
    make_sandbox_progress_handler,
    sandbox,
)

logger = get_logger("adapters.sqlite")

_ALLOWED_PRAGMAS = ALLOWED_PRAGMAS


def make_authorizer() -> SandboxAuthorizer:
    """SQLite authorizer blocking ATTACH, DETACH, dangerous PRAGMAs, load_extension."""
    return make_sandbox_authorizer()


def make_progress_handler(deadline_s: float = 2.0):
    """SQLite progress handler aborting queries running longer than deadline."""
    return make_sandbox_progress_handler(deadline_s)


def install_sandbox(conn: sqlite3.Connection, timeout_s: float = 2.0) -> SandboxAuthorizer:
    """Install authorizer and progress handler on an untrusted execution connection."""
    auth = make_sandbox_authorizer()
    conn.set_authorizer(auth)
    conn.set_progress_handler(make_sandbox_progress_handler(timeout_s), 10_000)
    return auth


class SQLiteAdapter(DbAdapter):
    """Adapter for SQLite sources."""

    def __init__(
        self,
        source_path: str | Path | None = None,
        source_bytes: bytes | None = None,
        sql_dump: str | None = None,
        source_conn: sqlite3.Connection | None = None,
    ):
        self.settings = get_settings()
        self.source_path = Path(source_path) if source_path else None
        self.source_bytes = source_bytes
        self.sql_dump = sql_dump
        self._source_conn: sqlite3.Connection | None = source_conn
        self._initial_hash: str = ""
        self._is_temp_file: bool = False

    @classmethod
    def from_file(cls, path: str | Path) -> "SQLiteAdapter":
        """Create adapter from SQLite file path."""
        return cls(source_path=path)

    @classmethod
    def from_bytes(cls, data: bytes) -> "SQLiteAdapter":
        """Create adapter from in-memory database bytes."""
        return cls(source_bytes=data)

    @classmethod
    def from_sql_dump(cls, sql: str) -> "SQLiteAdapter":
        """Create adapter from SQL schema/data dump string."""
        return cls(sql_dump=sql)

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection) -> "SQLiteAdapter":
        """Create adapter from existing sqlite3 connection."""
        return cls(source_conn=conn)

    @classmethod
    def create_in_memory(cls) -> "SQLiteAdapter":
        """Create adapter with clean in-memory database."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        return cls(source_conn=conn)

    def open_readonly(self) -> sqlite3.Connection:
        """Open source database in read-only mode and return connection."""
        if self._source_conn is not None:
            self._initial_hash = self.source_bytes_hash()
            return self._source_conn

        # 1. From SQL dump script
        if self.sql_dump is not None:
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            install_sandbox(conn, timeout_s=self.settings.sandbox_timeout_s)
            try:
                conn.executescript(self.sql_dump)
            except sqlite3.Error as exc:
                conn.close()
                msg = str(exc)
                if "not authorized" in msg.lower() or "authorization denied" in msg.lower():
                    raise AgentError(f"SANDBOX VIOLATION in schema dump: {msg}") from exc
                if "interrupted" in msg.lower():
                    raise AgentError("TIMEOUT in schema dump execution.") from exc
                raise AgentError(f"Schema SQL error: {exc}") from exc
            conn.set_progress_handler(None, 0)
            self._source_conn = conn
            self._initial_hash = self.source_bytes_hash()
            return self._source_conn

        # 2. From raw bytes (e.g. uploaded via Streamlit)
        if self.source_bytes is not None:
            size_mb = len(self.source_bytes) / (1024 * 1024)
            if size_mb > self.settings.db_size_cap_mb:
                raise AgentError(
                    f"Uploaded database ({size_mb:.1f} MB) exceeds maximum allowed size "
                    f"of {self.settings.db_size_cap_mb} MB."
                )
            # Create in-memory DB and populate with deserialize
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            conn.deserialize(self.source_bytes)
            self._source_conn = conn
            self._initial_hash = hashlib.sha256(self.source_bytes).hexdigest()
            return self._source_conn

        # 3. From file path
        if self.source_path is not None:
            if not self.source_path.exists():
                raise AgentError(f"Database file not found: {self.source_path}")

            size_mb = self.source_path.stat().st_size / (1024 * 1024)
            if size_mb > self.settings.db_size_cap_mb:
                raise AgentError(
                    f"Database file '{self.source_path.name}' ({size_mb:.1f} MB) "
                    f"exceeds maximum allowed size of {self.settings.db_size_cap_mb} MB."
                )

            # Record initial byte hash of source file
            with open(self.source_path, "rb") as f:
                self._initial_hash = hashlib.sha256(f.read()).hexdigest()

            # Open with mode=ro URI to physically guarantee zero writes
            abs_path = self.source_path.resolve()
            uri = f"file:{abs_path.as_posix()}?mode=ro"
            try:
                self._source_conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            except sqlite3.Error as exc:
                raise AgentError(f"Failed to open database '{self.source_path.name}': {exc}") from exc
            return self._source_conn

        raise AgentError("No valid SQLite source provided (path, bytes, dump, or connection).")

    def source_bytes_hash(self) -> str:
        """Compute cryptographic hash of source database."""
        if self.source_path and self.source_path.exists():
            with open(self.source_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        if self._source_conn:
            try:
                raw = self._source_conn.serialize()
                return hashlib.sha256(raw).hexdigest()
            except Exception:
                pass
        return self._initial_hash

    def introspect(self) -> DatabaseSchema:
        """Introspect tables, columns, PKs, FKs, indexes, views, and triggers."""
        if not self._source_conn:
            self.open_readonly()
        conn = self._source_conn
        cur = conn.cursor()

        # Check initial foreign keys status
        cur.execute("PRAGMA foreign_keys")
        fk_val = cur.fetchone()
        initial_fk = bool(fk_val[0]) if fk_val else True

        # Tables
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        table_names = [r[0] for r in cur.fetchall()]

        tables: dict[str, TableSchema] = {}
        for tname in table_names:
            # Columns
            cur.execute(f"PRAGMA table_info(\"{tname}\")")
            cols: list[ColumnInfo] = []
            for r in cur.fetchall():
                # (cid, name, type, notnull, dflt_value, pk)
                cols.append(
                    ColumnInfo(
                        name=r[1],
                        type=r[2] or "TEXT",
                        notnull=bool(r[3]),
                        default=str(r[4]) if r[4] is not None else None,
                        pk=r[5],
                    )
                )

            # Foreign keys
            cur.execute(f"PRAGMA foreign_key_list(\"{tname}\")")
            fks: list[ForeignKeyInfo] = []
            for r in cur.fetchall():
                # (id, seq, table, from, to, on_update, on_delete, match)
                fks.append(
                    ForeignKeyInfo(
                        from_col=r[3],
                        to_table=r[2],
                        to_col=r[4] or "",
                        on_update=r[5] or "NO ACTION",
                        on_delete=r[6] or "NO ACTION",
                    )
                )

            # Indexes
            cur.execute(f"PRAGMA index_list(\"{tname}\")")
            idxs: list[IndexInfo] = []
            for r in cur.fetchall():
                # (seq, name, unique, origin, partial)
                idx_name = r[1]
                idx_unique = bool(r[2])
                cur.execute(f"PRAGMA index_info(\"{idx_name}\")")
                idx_cols = [ir[2] for ir in cur.fetchall()]

                cur.execute(
                    "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
                    (idx_name,),
                )
                sql_row = cur.fetchone()
                idx_sql = sql_row[0] if sql_row else None

                idxs.append(
                    IndexInfo(
                        name=idx_name,
                        table=tname,
                        unique=idx_unique,
                        sql=idx_sql,
                        columns=idx_cols,
                    )
                )

            # Row count
            try:
                cur.execute(f"SELECT COUNT(*) FROM \"{tname}\"")
                row_cnt = cur.fetchone()[0]
            except Exception:
                row_cnt = 0

            tables[tname] = TableSchema(
                name=tname,
                columns=cols,
                foreign_keys=fks,
                indexes=idxs,
                row_count=row_cnt,
            )

        # Views
        cur.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='view' ORDER BY name"
        )
        views = {r[0]: ViewInfo(name=r[0], sql=r[1]) for r in cur.fetchall()}

        # Triggers
        cur.execute(
            "SELECT name, tbl_name, sql FROM sqlite_master WHERE type='trigger' ORDER BY name"
        )
        triggers = {
            r[0]: TriggerInfo(name=r[0], table=r[1], sql=r[2])
            for r in cur.fetchall()
        }

        return DatabaseSchema(
            dialect="sqlite",
            tables=tables,
            views=views,
            triggers=triggers,
            initial_foreign_keys=initial_fk,
        )

    def profile(self, table: str) -> dict:
        """Generate summary statistics for table columns without leaking full data."""
        if not self._source_conn:
            self.open_readonly()
        conn = self._source_conn
        cur = conn.cursor()

        cur.execute(f"PRAGMA table_info(\"{table}\")")
        cols = [r[1] for r in cur.fetchall()]

        cur.execute(f"SELECT COUNT(*) FROM \"{table}\"")
        total_rows = cur.fetchone()[0]

        profile_data: dict[str, dict] = {}
        for col in cols:
            cur.execute(
                f"SELECT COUNT(*) FROM \"{table}\" WHERE \"{col}\" IS NULL"
            )
            null_count = cur.fetchone()[0]
            cur.execute(
                f"SELECT COUNT(DISTINCT \"{col}\") FROM \"{table}\""
            )
            distinct_count = cur.fetchone()[0]

            profile_data[col] = {
                "null_count": null_count,
                "null_rate": round(null_count / total_rows, 4) if total_rows else 0.0,
                "distinct_count": distinct_count,
                "total_rows": total_rows,
            }

        return {"table": table, "total_rows": total_rows, "columns": profile_data}

    def clone(self) -> sqlite3.Connection:
        """Create disposable in-RAM clone in <5ms."""
        if not self._source_conn:
            self.open_readonly()
        clone_conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._source_conn.backup(clone_conn)
        if getattr(self, "inject_traps", False):
            try:
                from timemachine.traps.generator import inject_adversarial_traps
                schema = self.introspect()
                inject_adversarial_traps(clone_conn, schema)
            except Exception as exc:
                logger.debug("Trap injection notice on clone: %s", exc)
        return clone_conn

    def run_in_sandbox(
        self,
        clone_conn: sqlite3.Connection,
        sql: str,
        fk_enabled: bool = True,
    ) -> tuple[bool, str | None, float]:
        total_rows = 0
        try:
            cur_count = clone_conn.cursor()
            cur_count.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            for tbl_row in cur_count.fetchall():
                try:
                    cur_count.execute(f'SELECT COUNT(*) FROM "{tbl_row[0]}"')
                    total_rows += cur_count.fetchone()[0]
                except Exception:
                    pass
        except Exception:
            pass

        scaled_timeout = calc_sandbox_timeout(total_rows, base_timeout_s=self.settings.sandbox_timeout_s)
        cur = clone_conn.cursor()
        t0 = time.perf_counter()
        authorizer = None

        try:
            with sandbox(clone_conn, timeout_s=scaled_timeout) as auth:
                authorizer = auth
                # Set FK state for test
                cur.execute(f"PRAGMA foreign_keys = {'ON' if fk_enabled else 'OFF'}")
                clone_conn.executescript(sql)
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                return True, None, elapsed_ms
        except sqlite3.Error as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            msg = str(exc)
            if "interrupted" in msg.lower():
                return False, f"TIMEOUT after {scaled_timeout:.1f}s: query exceeded execution time limit.", elapsed_ms
            if "not authorized" in msg.lower() or "authorization denied" in msg.lower():
                denial_reason = authorizer.last_denial if authorizer and authorizer.last_denial else msg
                return False, f"SANDBOX VIOLATION: {denial_reason}", elapsed_ms
            return False, f"SQLITE EXECUTION CRASH: {msg}", elapsed_ms

    def close(self) -> None:
        """Close source connection."""
        if self._source_conn:
            try:
                self._source_conn.close()
            except Exception:
                pass
            self._source_conn = None
