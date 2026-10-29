"""SQLite 12-step table rebuild procedure and rollback verification."""

import sqlite3
from typing import Any

from timemachine.logging import get_logger
from timemachine.verify.sandbox import (
    apply_sandbox,
    calc_sandbox_timeout,
    make_sandbox_progress_handler,
    quote_ident,
)

logger = get_logger("verify.rebuild")


class RollbackVerificationResult(dict):
    """Result of rollback roundtrip verification.

    Inherits from dict so caller can access result["classification"],
    result["lossy_columns"], result["sample_diffs"], and unpacks as
    (rollback_verified, error, lossy_columns) for backwards compatibility.
    """

    def __init__(
        self,
        classification: str,
        lossy_columns: list[str],
        sample_diffs: list[dict[str, Any]],
        error: str | None = None,
    ):
        verified = classification in ("EXACT", "EQUIVALENT_AFTER_NORMALIZATION")
        super().__init__(
            classification=classification,
            lossy_columns=lossy_columns,
            sample_diffs=sample_diffs,
            rollback_verified=verified,
            error=error,
        )

    @property
    def classification(self) -> str:
        return self["classification"]

    @property
    def lossy_columns(self) -> list[str]:
        return self["lossy_columns"]

    @property
    def sample_diffs(self) -> list[dict[str, Any]]:
        return self["sample_diffs"]

    @property
    def rollback_verified(self) -> bool:
        return self["rollback_verified"]

    @property
    def error(self) -> str | None:
        return self["error"]

    def __iter__(self):
        yield self.rollback_verified
        yield self.error
        yield self.lossy_columns


def verify_rollback_roundtrip(
    base_conn: sqlite3.Connection,
    up_sql: str,
    down_sql: str,
    target_table: str,
    fk_enabled: bool = True,
    base_timeout_s: float = 2.0,
) -> RollbackVerificationResult:
    """Verify that applying up_sql then down_sql returns the database to its exact baseline state.

    Compares per row by primary key (or composite PK / rowid) and per column by name.
    Classifies result as: EXACT, EQUIVALENT_AFTER_NORMALIZATION, LOSSY, or BROKEN.
    """
    try:
        base_conn.commit()
    except Exception:
        pass

    clone = sqlite3.connect(":memory:", check_same_thread=False)
    base_conn.backup(clone)
    cur = clone.cursor()

    safe_table = quote_ident(target_table)

    # Read baseline table structure and rows
    b_cur = base_conn.cursor()
    b_cur.execute(f"PRAGMA table_info({safe_table})")
    b_col_info = b_cur.fetchall()
    if not b_col_info:
        clone.close()
        return RollbackVerificationResult(
            classification="BROKEN",
            lossy_columns=[],
            sample_diffs=[],
            error=f"Target table '{target_table}' not found in baseline schema.",
        )

    b_cols = [r[1] for r in b_col_info]
    pk_cols = [r[1] for r in b_col_info if r[5] > 0]
    quoted_cols = ", ".join(quote_ident(c) for c in b_cols)

    orig_rows_by_pk: dict[Any, dict[str, Any]] = {}
    if not pk_cols:
        b_cur.execute(f"SELECT rowid, {quoted_cols} FROM {safe_table}")
        b_rows = b_cur.fetchall()
        for r in b_rows:
            pk_val = str(r[0])
            orig_rows_by_pk[pk_val] = {col: r[i + 1] for i, col in enumerate(b_cols)}
    else:
        b_cur.execute(f"SELECT {quoted_cols} FROM {safe_table}")
        b_rows = b_cur.fetchall()
        for r in b_rows:
            row_dict = {col: r[i] for i, col in enumerate(b_cols)}
            pk_val = tuple(row_dict[c] for c in pk_cols) if len(pk_cols) > 1 else str(row_dict[pk_cols[0]])
            orig_rows_by_pk[pk_val] = row_dict

    total_rows = len(orig_rows_by_pk)
    try:
        timeout_s = float(base_timeout_s) if isinstance(base_timeout_s, (int, float)) else 2.0
    except (TypeError, ValueError):
        timeout_s = 2.0
    scaled_timeout = calc_sandbox_timeout(total_rows, base_timeout_s=timeout_s)
    apply_sandbox(clone, deadline_s=scaled_timeout)

    # Enforce foreign keys setting
    cur.execute(f"PRAGMA foreign_keys = {'ON' if fk_enabled else 'OFF'};")

    try:
        # Step 1: Apply UP migration
        clone.executescript(up_sql)
    except Exception as exc:
        msg = str(exc)
        if "interrupted" in msg.lower():
            err_msg = f"TIMEOUT after {scaled_timeout:.1f}s during forward migration execution in roundtrip."
        elif "not authorized" in msg.lower() or "authorization denied" in msg.lower():
            err_msg = f"SANDBOX VIOLATION during forward migration in roundtrip: {exc}"
        else:
            err_msg = f"FORWARD_MIGRATION_EXECUTION_ERROR in UP script: {exc}"
        clone.close()
        return RollbackVerificationResult(
            classification="BROKEN",
            lossy_columns=[],
            sample_diffs=[],
            error=err_msg,
        )

    try:
        # Refresh timeout handler for DOWN migration
        clone.set_progress_handler(make_sandbox_progress_handler(scaled_timeout), 10_000)

        # Step 2: Apply DOWN migration
        clone.executescript(down_sql)

        # Step 3: Check restored table schema
        cur.execute(f"PRAGMA table_info({safe_table})")
        down_col_info = cur.fetchall()
        if not down_col_info:
            return RollbackVerificationResult(
                classification="BROKEN",
                lossy_columns=b_cols,
                sample_diffs=[],
                error=f"Target table '{target_table}' was not restored by rollback script.",
            )

        down_cols = [r[1] for r in down_col_info]
        missing_cols = [c for c in b_cols if c not in down_cols]
        if missing_cols:
            return RollbackVerificationResult(
                classification="BROKEN",
                lossy_columns=missing_cols,
                sample_diffs=[],
                error=f"Rollback dropped columns: {', '.join(missing_cols)}",
            )

        # Step 4: Query restored rows using exact baseline columns
        restored_rows_by_pk: dict[Any, dict[str, Any]] = {}
        if not pk_cols:
            cur.execute(f"SELECT rowid, {quoted_cols} FROM {safe_table}")
            down_rows = cur.fetchall()
            for r in down_rows:
                pk_val = str(r[0])
                restored_rows_by_pk[pk_val] = {col: r[i + 1] for i, col in enumerate(b_cols)}
        else:
            cur.execute(f"SELECT {quoted_cols} FROM {safe_table}")
            down_rows = cur.fetchall()
            for r in down_rows:
                row_dict = {col: r[i] for i, col in enumerate(b_cols)}
                pk_val = tuple(row_dict[c] for c in pk_cols) if len(pk_cols) > 1 else str(row_dict[pk_cols[0]])
                restored_rows_by_pk[pk_val] = row_dict

        if set(orig_rows_by_pk.keys()) != set(restored_rows_by_pk.keys()):
            diff_keys = set(orig_rows_by_pk.keys()) ^ set(restored_rows_by_pk.keys())
            return RollbackVerificationResult(
                classification="BROKEN",
                lossy_columns=[],
                sample_diffs=[],
                error=(
                    f"Rollback row count mismatch: {len(orig_rows_by_pk)} baseline rows vs "
                    f"{len(restored_rows_by_pk)} restored rows (differing keys: {list(diff_keys)[:3]})."
                ),
            )

        # Step 5: Compare per row by primary key and per column by name
        has_whitespace_only_diff = False
        has_lossy_diff = False
        lossy_columns: list[str] = []
        sample_diffs: list[dict[str, Any]] = []

        for pk_val, orig_row in orig_rows_by_pk.items():
            rest_row = restored_rows_by_pk[pk_val]
            for col in b_cols:
                orig_val = orig_row.get(col)
                rest_val = rest_row.get(col)

                if orig_val == rest_val:
                    continue

                # Check if difference is only leading/trailing whitespace
                if isinstance(orig_val, str) and isinstance(rest_val, str) and orig_val.strip() == rest_val.strip():
                    has_whitespace_only_diff = True
                else:
                    has_lossy_diff = True
                    if col not in lossy_columns:
                        lossy_columns.append(col)
                    if len(sample_diffs) < 3:
                        sample_diffs.append({
                            "pk": str(pk_val),
                            "column": col,
                            "original": orig_val,
                            "restored": rest_val,
                        })

        if has_lossy_diff:
            diff_summary = "; ".join(
                [f"PK {d['pk']} col {d['column']}: {d['original']!r} -> {d['restored']!r}" for d in sample_diffs]
            )
            return RollbackVerificationResult(
                classification="LOSSY",
                lossy_columns=lossy_columns,
                sample_diffs=sample_diffs,
                error=f"Rollback modified values in columns {lossy_columns}: {diff_summary}",
            )

        if has_whitespace_only_diff:
            return RollbackVerificationResult(
                classification="EQUIVALENT_AFTER_NORMALIZATION",
                lossy_columns=[],
                sample_diffs=[],
                error=None,
            )

        return RollbackVerificationResult(
            classification="EXACT",
            lossy_columns=[],
            sample_diffs=[],
            error=None,
        )

    except Exception as exc:
        msg = str(exc)
        if "interrupted" in msg.lower():
            err_msg = f"TIMEOUT after {scaled_timeout:.1f}s during rollback execution."
        elif "not authorized" in msg.lower() or "authorization denied" in msg.lower():
            err_msg = f"SANDBOX VIOLATION during rollback: {exc}"
        else:
            err_msg = f"ROLLBACK_EXECUTION_ERROR: {exc}"
        return RollbackVerificationResult(
            classification="BROKEN",
            lossy_columns=[],
            sample_diffs=[],
            error=err_msg,
        )
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
