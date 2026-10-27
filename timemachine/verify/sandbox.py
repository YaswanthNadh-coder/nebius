"""Sandboxing utilities and identifier validation."""

import re
import sqlite3
import time

from timemachine.config import AgentError

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_SQLITE_OK = 0
_SQLITE_DENY = 1
_SQLITE_PRAGMA = 19
_SQLITE_ATTACH = 24
_SQLITE_DETACH = 25
_SQLITE_FUNCTION = 31

_ALLOWED_PRAGMAS = frozenset({
    "table_info",
    "table_xinfo",
    "foreign_key_check",
    "foreign_key_list",
    "integrity_check",
    "foreign_keys",
    "index_list",
    "index_info",
})


def quote_ident(name: str) -> str:
    """Validate and double-quote a SQL identifier against ^[A-Za-z_][A-Za-z0-9_]*$."""
    if not isinstance(name, str) or not IDENTIFIER_RE.match(name):
        raise AgentError(f"Invalid SQL identifier: {name!r}")
    return f'"{name}"'


def make_sandbox_authorizer():
    """SQLite authorizer callback denying ATTACH, DETACH, dangerous PRAGMAs, and load_extension."""
    def _auth(action, arg1, _arg2, _db_name, _trigger_name):
        if action in (_SQLITE_ATTACH, _SQLITE_DETACH):
            return _SQLITE_DENY
        if action == _SQLITE_PRAGMA:
            if arg1 and arg1.lower() not in _ALLOWED_PRAGMAS:
                return _SQLITE_DENY
        if action == _SQLITE_FUNCTION:
            if arg1 and arg1.lower() == "load_extension":
                return _SQLITE_DENY
        return _SQLITE_OK
    return _auth


def make_sandbox_progress_handler(deadline_s: float = 2.0):
    """SQLite progress handler callback returning 1 after deadline_s passes."""
    t0 = time.monotonic()
    def _tick():
        return 1 if (time.monotonic() - t0) > deadline_s else 0
    return _tick


def apply_sandbox(conn: sqlite3.Connection, deadline_s: float = 2.0, timeout_sec: float | None = None) -> None:
    """Apply sandbox authorizer and progress handler to a sqlite connection."""
    limit = timeout_sec if timeout_sec is not None else deadline_s
    conn.set_authorizer(make_sandbox_authorizer())
    conn.set_progress_handler(make_sandbox_progress_handler(limit), 10_000)
