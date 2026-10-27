"""SQLite 12-step table rebuild procedure and rollback verification."""

import hashlib
import sqlite3
from typing import Any

from timemachine.logging import get_logger
from timemachine.verify.sandbox import apply_sandbox

logger = get_logger("verify.rebuild")


def verify_rollback_roundtrip(
    base_conn: sqlite3.Connection,
    up_sql: str,
    down_sql: str,
    target_table: str,
) -> tuple[bool, str | None, list[str]]:
    """Verify that applying up_sql then down_sql returns the database to its exact baseline state."""
    clone = sqlite3.connect(":memory:", check_same_thread=False)
    base_conn.backup(clone)
    apply_sandbox(clone)
    cur = clone.cursor()

    # Compute baseline checksum for target table
    b_cur = base_conn.cursor()
    b_cur.execute(f"SELECT * FROM \"{target_table}\"")
    b_rows = b_cur.fetchall()
    orig_checksum = hashlib.sha256(str(b_rows).encode()).hexdigest()

    lossy_columns: list[str] = []

    try:
        # Step 1: Apply UP migration
        clone.executescript(up_sql)

        # Step 2: Apply DOWN migration
        clone.executescript(down_sql)

        # Step 3: Compare post-rollback state to baseline
        cur.execute(f"SELECT * FROM \"{target_table}\"")
        down_rows = cur.fetchall()
        down_checksum = hashlib.sha256(str(down_rows).encode()).hexdigest()

        if orig_checksum != down_checksum:
            # Check which columns lost data
            cur.execute(f"PRAGMA table_info(\"{target_table}\")")
            down_cols = [r[1] for r in cur.fetchall()]
            b_cur.execute(f"PRAGMA table_info(\"{target_table}\")")
            b_cols = [r[1] for r in b_cur.fetchall()]

            for bc in b_cols:
                if bc not in down_cols:
                    lossy_columns.append(bc)

            return False, "ROUNDTRIP_CHECKSUM_MISMATCH: Rollback did not fully restore original row state.", lossy_columns

        return True, None, []
    except Exception as exc:
        return False, f"ROLLBACK_EXECUTION_ERROR: {exc}", []
    finally:
        clone.close()


def estimate_production_lock_time(
    clone_elapsed_ms: float,
    clone_rows: int,
    production_rows: int,
) -> dict[str, Any]:
    """Extrapolate lock time from clone execution duration to full production size."""
    if clone_rows <= 0:
        return {
            "estimated_lock_ms": clone_elapsed_ms,
            "estimated_lock_sec": round(clone_elapsed_ms / 1000, 2),
            "risk_level": "LOW",
            "warning": None,
        }

    ratio = max(1.0, production_rows / clone_rows)
    # Estimate with slight O(N log N) scaling factor for index rebuilds
    scaled_ms = clone_elapsed_ms * ratio * 1.15
    scaled_sec = round(scaled_ms / 1000, 2)

    risk_level = "LOW"
    warning = None
    if scaled_sec > 10.0:
        risk_level = "HIGH"
        warning = f"High lock risk: Estimated migration lock duration of {scaled_sec}s may stall production traffic."
    elif scaled_sec > 2.0:
        risk_level = "MEDIUM"
        warning = f"Moderate lock risk: Estimated migration lock duration of {scaled_sec}s."

    return {
        "estimated_lock_ms": round(scaled_ms, 2),
        "estimated_lock_sec": scaled_sec,
        "risk_level": risk_level,
        "warning": warning,
    }
