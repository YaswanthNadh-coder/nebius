"""Sandboxing utilities and identifier validation."""

import re
import sqlite3
import time
from typing import Any

from timemachine.config import AgentError

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_SQLITE_OK = 0
_SQLITE_DENY = 1
_SQLITE_PRAGMA = 19
_SQLITE_ATTACH = 24
_SQLITE_DETACH = 25
_SQLITE_VACUUM = 28
_SQLITE_FUNCTION = 31

ALLOWED_PRAGMAS: frozenset[str] = frozenset({
    "defer_foreign_keys",
    "legacy_alter_table",
    "recursive_triggers",
    "user_version",
    "foreign_keys",
    "foreign_key_check",
    "table_info",
    "table_xinfo",
    "index_list",
    "index_info",
    "foreign_key_list",
    "integrity_check",
})
_ALLOWED_PRAGMAS = ALLOWED_PRAGMAS


def quote_ident(name: str) -> str:
    """Validate and double-quote a SQL identifier against ^[A-Za-z_][A-Za-z0-9_]*$."""
    if not isinstance(name, str) or not IDENTIFIER_RE.match(name):
        raise AgentError(f"Invalid SQL identifier: {name!r}")
    return f'"{name}"'


def calc_sandbox_timeout(row_count: int, base_timeout_s: float = 2.0) -> float:
    """Scale sandbox_timeout_s with row count: max(2s, 2s + 1s per 100k rows), capped at 30s."""
    scaled = max(base_timeout_s, base_timeout_s + (max(0, row_count) / 100_000.0))
    return min(30.0, scaled)


class SandboxAuthorizer:
    """SQLite authorizer callback denying ATTACH, DETACH, dangerous PRAGMAs, VACUUM, and load_extension.

    Records the denied action (action code name, arg1, arg2) and provides an informative denial reason.
    """

    def __init__(self, allowed_pragmas: frozenset[str] = ALLOWED_PRAGMAS):
        self.allowed_pragmas = allowed_pragmas
        self.last_denial: str | None = None
        self.last_denial_details: dict[str, Any] = {}

    def __call__(
        self,
        action: int,
        arg1: str | None,
        arg2: str | None,
        db_name: str | None,
        trigger_name: str | None,
    ) -> int:
        if action == _SQLITE_ATTACH:
            # In SQLite, VACUUM triggers ATTACH with arg1 == ""
            if not arg1:
                self.last_denial = "VACUUM is not permitted."
                self.last_denial_details = {"action": "VACUUM", "arg1": arg1, "arg2": arg2}
                return _SQLITE_DENY
            self.last_denial = f"ATTACH is not permitted (attempted {arg1})."
            self.last_denial_details = {"action": "ATTACH", "arg1": arg1, "arg2": arg2}
            return _SQLITE_DENY

        if action == _SQLITE_DETACH:
            self.last_denial = f"DETACH is not permitted (attempted {arg1})." if arg1 else "DETACH is not permitted."
            self.last_denial_details = {"action": "DETACH", "arg1": arg1, "arg2": arg2}
            return _SQLITE_DENY

        if action == _SQLITE_VACUUM:
            self.last_denial = "VACUUM is not permitted."
            self.last_denial_details = {"action": "VACUUM", "arg1": arg1, "arg2": arg2}
            return _SQLITE_DENY

        if action == _SQLITE_PRAGMA:
            pragma_name = (arg1 or "").lower()
            if pragma_name not in self.allowed_pragmas:
                permitted_str = ", ".join(sorted(self.allowed_pragmas))
                self.last_denial = f"PRAGMA {arg1} is not permitted. Permitted PRAGMAs: {permitted_str}"
                self.last_denial_details = {"action": "PRAGMA", "arg1": arg1, "arg2": arg2}
                return _SQLITE_DENY

        if action == _SQLITE_FUNCTION:
            fn_name = (arg2 or arg1 or "").lower()
            if fn_name == "load_extension":
                self.last_denial = "load_extension is not permitted."
                self.last_denial_details = {"action": "FUNCTION", "arg1": arg1, "arg2": arg2}
                return _SQLITE_DENY

        return _SQLITE_OK


def make_sandbox_authorizer(allowed_pragmas: frozenset[str] = ALLOWED_PRAGMAS) -> SandboxAuthorizer:
    """Return a SandboxAuthorizer callable."""
    return SandboxAuthorizer(allowed_pragmas=allowed_pragmas)


def make_sandbox_progress_handler(deadline_s: float = 2.0):
    """SQLite progress handler callback returning 1 after deadline_s passes."""
    t0 = time.monotonic()

    def _tick():
        return 1 if (time.monotonic() - t0) > deadline_s else 0

    return _tick


def apply_sandbox(
    conn: sqlite3.Connection,
    deadline_s: float = 2.0,
    timeout_sec: float | None = None,
) -> SandboxAuthorizer:
    """Apply sandbox authorizer and progress handler to a sqlite connection."""
    limit = timeout_sec if timeout_sec is not None else deadline_s
    auth = make_sandbox_authorizer()
    conn.set_authorizer(auth)
    conn.set_progress_handler(make_sandbox_progress_handler(limit), 10_000)
    return auth
