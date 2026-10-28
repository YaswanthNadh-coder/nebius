"""Tests for Item B: Sandbox Authorizer (permitted/denied PRAGMAs, action names, dynamic timeout scaling)."""

import sqlite3

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.verify.sandbox import (
    ALLOWED_PRAGMAS,
    calc_sandbox_timeout,
    make_sandbox_authorizer,
)


def test_each_allowed_pragma_passes():
    """All permitted PRAGMAs on the in-memory clone succeed without denial."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT);")
    conn.commit()

    authorizer = make_sandbox_authorizer()
    conn.set_authorizer(authorizer)

    allowed = [
        "PRAGMA defer_foreign_keys = ON;",
        "PRAGMA legacy_alter_table = ON;",
        "PRAGMA recursive_triggers = ON;",
        "PRAGMA user_version = 42;",
        "PRAGMA foreign_keys = ON;",
        "PRAGMA foreign_key_check;",
        "PRAGMA table_info(t);",
        "PRAGMA table_xinfo(t);",
        "PRAGMA index_list(t);",
        "PRAGMA index_info('sqlite_autoindex_t_1');",
        "PRAGMA foreign_key_list(t);",
        "PRAGMA integrity_check;",
    ]

    for pragma_sql in allowed:
        cur = conn.cursor()
        cur.execute(pragma_sql)
        cur.fetchall()

    assert authorizer.last_denial is None
    conn.close()


def test_denied_actions_return_messages_naming_them():
    """Each denied action (PRAGMA, VACUUM, ATTACH, DETACH, load_extension) returns a message naming it."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.clone()
    conn.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT);")
    conn.commit()

    # 1. PRAGMA writable_schema
    ok, err, _ = adapter.run_in_sandbox(conn, "PRAGMA writable_schema = ON;")
    assert not ok
    assert "SANDBOX VIOLATION" in err
    assert "PRAGMA writable_schema is not permitted" in err
    assert "Permitted PRAGMAs:" in err

    # 2. PRAGMA journal_mode
    ok, err, _ = adapter.run_in_sandbox(conn, "PRAGMA journal_mode = WAL;")
    assert not ok
    assert "SANDBOX VIOLATION" in err
    assert "PRAGMA journal_mode is not permitted" in err
    assert "Permitted PRAGMAs:" in err

    # 3. VACUUM
    ok, err, _ = adapter.run_in_sandbox(conn, "VACUUM;")
    assert not ok
    assert "SANDBOX VIOLATION" in err
    assert "VACUUM" in err

    # 4. ATTACH
    ok, err, _ = adapter.run_in_sandbox(conn, "ATTACH DATABASE ':memory:' AS rogue;")
    assert not ok
    assert "SANDBOX VIOLATION" in err
    assert "ATTACH" in err

    # 5. DETACH
    ok, err, _ = adapter.run_in_sandbox(conn, "DETACH DATABASE rogue;")
    assert not ok
    assert "SANDBOX VIOLATION" in err
    assert "DETACH" in err

    # 6. load_extension
    ok, err, _ = adapter.run_in_sandbox(conn, "SELECT load_extension('hack');")
    assert not ok
    assert "SANDBOX VIOLATION" in err
    assert "load_extension" in err

    conn.close()


def test_calc_sandbox_timeout_scaling():
    """Scale sandbox_timeout_s with row count: max(2s, 2s + 1s per 100k rows), capped at 30s."""
    assert calc_sandbox_timeout(0) == 2.0
    assert calc_sandbox_timeout(50_000) == 2.5
    assert calc_sandbox_timeout(100_000) == 3.0
    assert calc_sandbox_timeout(200_000) == 4.0
    assert calc_sandbox_timeout(1_000_000) == 12.0
    assert calc_sandbox_timeout(3_000_000) == 30.0  # capped at 30s
    assert calc_sandbox_timeout(10_000_000) == 30.0  # capped at 30s


def test_allowed_pragmas_contains_all_12_required():
    """Verify ALLOWED_PRAGMAS set contains exactly the 12 permitted pragmas."""
    expected = {
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
    }
    assert ALLOWED_PRAGMAS == expected
