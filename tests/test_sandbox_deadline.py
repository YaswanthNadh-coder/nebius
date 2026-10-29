"""Tests for Item E: Sandbox deadline leak and per-execution sandbox context manager."""

import sqlite3
import time

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.verify.sandbox import sandbox


def test_sandbox_context_manager_clears_handler_on_exit():
    """Execute in sandbox, sleep 2.3s, then run a query on same connection (must succeed)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (x INTEGER);")
    conn.execute("INSERT INTO t VALUES (1);")

    # Use sandbox context manager with 1.0s timeout
    with sandbox(conn, timeout_s=1.0):
        conn.execute("INSERT INTO t VALUES (2);")

    # Sleep past the deadline
    time.sleep(2.3)

    # Run heavy recursive CTE query on same connection; must not raise 'interrupted'
    cur = conn.cursor()
    cur.execute("""
        WITH RECURSIVE cnt(x) AS (
            SELECT 1
            UNION ALL
            SELECT x + 1 FROM cnt WHERE x < 50000
        )
        SELECT COUNT(*) FROM cnt;
    """)
    row = cur.fetchone()
    assert row[0] == 50000


def test_run_in_sandbox_does_not_leak_deadline_to_subsequent_queries():
    """run_in_sandbox on adapter must not leave an active progress handler that interrupts subsequent queries."""
    adapter = SQLiteAdapter.create_in_memory()
    conn = adapter.clone()

    ok, err, _ = adapter.run_in_sandbox(conn, "CREATE TABLE test_tbl (id INTEGER);", fk_enabled=True)
    assert ok is True

    # Sleep past default sandbox timeout (2.0s -> sleep 2.3s)
    time.sleep(2.3)

    # Running another query on the same connection must succeed
    cur = conn.cursor()
    cur.execute("""
        WITH RECURSIVE cnt(x) AS (
            SELECT 1
            UNION ALL
            SELECT x + 1 FROM cnt WHERE x < 50000
        )
        SELECT COUNT(*) FROM cnt;
    """)
    row = cur.fetchone()
    assert row[0] == 50000
