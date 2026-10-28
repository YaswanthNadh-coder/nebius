"""Local SQLite persistence for run logs, spec hashes, and audit telemetry."""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any

from timemachine.agent.loop import AgentRunResult

DB_PATH = os.path.join(os.path.expanduser("~"), ".timemachine", "runs.db")


def _init_history_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Ensure directory and schema exist for local run history database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS run_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            target_table TEXT NOT NULL,
            goal TEXT NOT NULL,
            status TEXT NOT NULL,
            up_sql TEXT,
            down_sql TEXT,
            spec_json TEXT,
            steps_count INTEGER,
            latency_ms REAL,
            rollback_verified INTEGER,
            report_json TEXT
        )
    """)
    conn.commit()
    return conn


def save_run_history(
    result: AgentRunResult,
    target_table: str,
    goal: str,
    db_path: str = DB_PATH,
) -> int:
    """Save an agent run result to the local SQLite run history."""
    conn = _init_history_db(db_path)
    try:
        cur = conn.cursor()
        now = datetime.now().isoformat()
        spec_json = result.spec.model_dump_json() if result.spec else "{}"
        report_data = {
            "status": result.status,
            "steps": [s.model_dump() for s in result.steps],
            "attempts": [a.model_dump() for a in getattr(result, "attempts", [])],
            "lock_estimate": result.lock_estimate,
            "lossy_columns": result.lossy_columns,
            "rollback_error": result.rollback_error,
            "stop_reason": result.stop_reason,
            "invariant_result": result.invariant_result.model_dump() if result.invariant_result else None,
        }
        cur.execute(
            """
            INSERT INTO run_history (
                timestamp, target_table, goal, status, up_sql, down_sql,
                spec_json, steps_count, latency_ms, rollback_verified, report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                target_table,
                goal,
                result.status,
                result.up_sql,
                result.down_sql,
                spec_json,
                len(result.steps),
                result.total_latency_ms,
                1 if result.rollback_verified else 0,
                json.dumps(report_data),
            ),
        )
        conn.commit()
        return cur.lastrowid or 0
    finally:
        conn.close()


def load_run_history(limit: int = 20, db_path: str = DB_PATH) -> list[dict[str, Any]]:
    """Retrieve historical runs ordered from newest to oldest."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM run_history ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()
