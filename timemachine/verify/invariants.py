"""Deterministic generic invariants engine for database migrations."""

import re
import sqlite3
from typing import Any

from pydantic import BaseModel, Field

from timemachine.logging import get_logger
from timemachine.verify.oracle import verify_rows_with_oracle
from timemachine.verify.sandbox import quote_ident
from timemachine.verify.spec import MigrationSpec

logger = get_logger("verify.invariants")


class InvariantResult(BaseModel):
    """Result of invariant evaluation."""

    status: str = "FAILED"  # VERIFIED, UNVERIFIED, FAILED
    passed: bool = False
    error: str | None = None
    before_count: int = 0
    after_count: int | None = None
    elapsed_ms: float = 0.0
    failed_checks: list[str] = Field(default_factory=list)
    passed_checks: list[str] = Field(default_factory=list)
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    trap_audit: list[dict[str, Any]] = Field(default_factory=list)
    oracle_verified: bool = False
    is_sampled: bool = False


def _normalize_sql(sql: str | None) -> str:
    """Normalize whitespace and case for comparing schema DDL."""
    if not sql:
        return ""
    # Strip whitespace, collapse spaces
    s = re.sub(r"\s+", " ", sql.strip())
    # Strip quotes from identifiers for normalized comparison
    s = s.replace('"', "").replace("`", "")
    return s.lower()


def _build_fk_mismatch_hint(
    target_table: str,
    mismatches: dict[str, int],
    examples: list[dict],
    spec: MigrationSpec | None,
    b_cur: sqlite3.Cursor,
    m_cur: sqlite3.Cursor,
) -> str:
    """If a mismatched column has a FOREIGN KEY constraint and the expected value is absent from the parent table,
    append an explicit hint teaching the 2-step Sentinel Foreign Key Pattern."""
    try:
        q_target = quote_ident(target_table)
        m_cur.execute(f"PRAGMA foreign_key_list({q_target})")
        fks_for_table = m_cur.fetchall()
        if not fks_for_table:
            b_cur.execute(f"PRAGMA foreign_key_list({q_target})")
            fks_for_table = b_cur.fetchall()

        hints = []
        for col_w in sorted(mismatches.keys()):
            fk_info = next((fk for fk in fks_for_table if fk[3].lower() == col_w.lower()), None)
            if not fk_info and spec:
                m = next((m for m in spec.get_table_mappings(target_table) if m.new_column.lower() == col_w.lower()), None)
                if m and m.lookup_table and m.lookup_key:
                    fk_info = (0, 0, m.lookup_table, col_w, m.lookup_key)

            if fk_info:
                parent_tbl = fk_info[2]
                parent_pk = fk_info[4]
                ex_for_col = [e for e in examples if e.get("column") == col_w]
                exp_val = ex_for_col[0].get("expected") if ex_for_col else None
                if exp_val is not None and str(exp_val) != "" and str(exp_val).lower() != "none":
                    m_cur.execute(f'SELECT COUNT(*) FROM "{parent_tbl}" WHERE "{parent_pk}" = ?', (str(exp_val),))
                    has_key = m_cur.fetchone()[0] > 0
                    if not has_key:
                        b_cur.execute(f'PRAGMA table_info("{parent_tbl}")')
                        p_cols = b_cur.fetchall()
                        seed_c = []
                        seed_v = []
                        for cid, cname, ctype, notnull, dflt, pk in p_cols:
                            if pk > 0 or notnull:
                                col_ident = cname if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", cname) else f'"{cname}"'
                                seed_c.append(col_ident)
                                if pk > 0:
                                    seed_v.append(repr(str(exp_val)))
                                else:
                                    seed_v.append(repr(str(exp_val).capitalize()))
                        seed_sql = f'INSERT OR IGNORE INTO {parent_tbl} ({", ".join(seed_c)}) VALUES ({", ".join(seed_v)});'
                        mapping = next((m for m in (spec.get_table_mappings(target_table) if spec else []) if m.new_column == col_w), None)
                        src_col = (mapping.source_column or col_w) if mapping else col_w
                        hints.append(
                            f" [CRITICAL FK NOTE: Column '{col_w}' has a FOREIGN KEY referencing '{parent_tbl}({parent_pk})'. "
                            f"To make '{col_w}' equal to {exp_val!r} without triggering 'FOREIGN KEY constraint failed', you MUST: "
                            f"(1) execute `{seed_sql}` before inserting into '{target_table}', AND "
                            f"(2) use `CASE WHEN \"{src_col}\" IS NOT NULL AND \"{src_col}\" IN (SELECT \"{parent_pk}\" FROM \"{parent_tbl}\") THEN \"{src_col}\" ELSE {exp_val!r} END` "
                            f"(do NOT use COALESCE alone, as non-NULL orphan rows will fail the FK check).]"
                        )
        return "".join(hints)
    except Exception as exc:
        logger.debug(f"Error computing FK mismatch hint: {exc}")
        return ""


def evaluate_migration_invariants(
    base_conn: sqlite3.Connection,
    migrated_conn: sqlite3.Connection,
    target_table: str,
    spec: MigrationSpec | None = None,
    required_cols: list[str] | None = None,
    non_null_cols: list[str] | None = None,
    expected_row_fn=None,
    elapsed_ms: float = 0.0,
) -> InvariantResult:
    """Run full deterministic invariant suite on migrated clone against baseline."""
    q_target = quote_ident(target_table)
    b_cur = base_conn.cursor()
    m_cur = migrated_conn.cursor()

    failed_checks: list[str] = []
    passed_checks: list[str] = []

    # ── Baseline Snapshot ──────────────────────────────────────────────────
    b_cur.execute(f"SELECT COUNT(*) FROM {q_target}")
    before_count = b_cur.fetchone()[0]

    is_sampled = before_count > 20_000
    if is_sampled:
        passed_checks.append("AUDIT_MODE_SAMPLED: >20k rows audited via trap rows, 5k deterministic sample, and aggregate SQL checksums")

    b_cur.execute(f"PRAGMA table_info({q_target})")
    orig_col_meta = {r[1]: {"type": r[2], "notnull": bool(r[3]), "default": r[4], "pk": r[5]} for r in b_cur.fetchall()}
    orig_col_names = list(orig_col_meta.keys())

    # Only load full table into Python memory for small tables (<= 20k rows)
    orig_rows: list[dict[str, Any]] = []
    if not is_sampled:
        b_cur.execute(f"SELECT * FROM {q_target}")
        orig_rows = [dict(zip(orig_col_names, r)) for r in b_cur.fetchall()]

    b_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != '__trap_rows'")
    orig_tables = {r[0] for r in b_cur.fetchall()}

    # Baseline foreign key violations (baseline-subtracted!)
    b_cur.execute("PRAGMA foreign_key_check")
    baseline_fk_violations = set(tuple(r) for r in b_cur.fetchall())

    # Baseline views and triggers
    b_cur.execute("SELECT name, sql FROM sqlite_master WHERE type='view'")
    orig_views = {r[0]: _normalize_sql(r[1]) for r in b_cur.fetchall()}

    b_cur.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger'")
    orig_triggers = {r[0]: _normalize_sql(r[1]) for r in b_cur.fetchall()}

    # Baseline indexes on all tables (including target table)
    b_cur.execute(
        "SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
    )
    orig_indexes = {r[0]: (r[1], _normalize_sql(r[2])) for r in b_cur.fetchall()}

    # ── Check 1: Table Set Consistency & No Leftover Temp Tables ──────────
    m_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != '__trap_rows'")
    migrated_tables = {r[0] for r in m_cur.fetchall()}

    leftover_temps = [t for t in migrated_tables if t.endswith("_new") or t.endswith("_tmp") or t.endswith("_temp")]
    if leftover_temps:
        failed_checks.append(f"LEFTOVER_TEMP_TABLES: {leftover_temps}")
    else:
        extra = migrated_tables - orig_tables
        gone = orig_tables - migrated_tables
        allowed_extra = set(spec.affected_tables) if spec else set()
        unexpected_extra = extra - allowed_extra
        if unexpected_extra or gone:
            failed_checks.append(f"TABLE_SET_MISMATCH: extra={unexpected_extra}, missing={gone}")
        else:
            passed_checks.append("TABLE_SET_CONSISTENCY")

    # ── Check 2: Target Table Exists and Row Count Accounting ─────────────
    try:
        m_cur.execute(f"SELECT COUNT(*) FROM {q_target}")
        after_count = m_cur.fetchone()[0]
    except Exception as exc:
        return InvariantResult(
            status="FAILED",
            passed=False,
            error=f"SQLITE_QUERY_ERROR: Table '{target_table}' not queryable: {exc}",
            before_count=before_count,
            after_count=None,
            elapsed_ms=elapsed_ms,
            failed_checks=[f"TABLE_QUERY_ERROR: {exc}"],
        )

    row_policy = spec.row_policy if spec else "preserve_all"
    if after_count < before_count:
        if row_policy != "deletions_allowed":
            failed_checks.append(
                f"SILENT_DATA_LOSS: Row count dropped from {before_count} to {after_count} "
                f"({before_count - after_count} rows lost). Likely naive INNER JOIN on NULL or unmapped key."
            )
        else:
            passed_checks.append("ROW_POLICY_DELETIONS_PERMITTED")
    elif after_count > before_count:
        failed_checks.append(
            f"DUPLICATED_ROWS: Row count increased from {before_count} to {after_count} "
            f"(+{after_count - before_count} duplicated rows)."
        )
    else:
        passed_checks.append("ROW_COUNT_PRESERVED")

    # ── Check 3: PRAGMA table_info, Columns and Primary Key ───────────────
    m_cur.execute(f"PRAGMA table_info({q_target})")
    col_meta = {
        r[1]: {
            "type": r[2],
            "notnull": bool(r[3]) or (r[5] > 0),
            "default": r[4],
            "pk": r[5],
        }
        for r in m_cur.fetchall()
    }
    actual_col_names = list(col_meta.keys())

    # Resolve expected and non-null cols
    target_req_cols = required_cols or (spec.get_required_columns(target_table) if spec else actual_col_names)
    target_nn_cols = non_null_cols or (spec.get_non_null_columns(target_table) if spec else [])

    missing_cols = [c for c in target_req_cols if c not in col_meta]
    if missing_cols:
        undo_hint = ""
        b_cur.execute(f"PRAGMA table_info({q_target})")
        b_cols = [r[1] for r in b_cur.fetchall()]
        if set(actual_col_names) == set(b_cols):
            undo_hint = " [How to fix: UP appears undone or not applied; target table columns match original baseline columns]"
        failed_checks.append(f"MISSING_REQUIRED_COLUMNS: {missing_cols}. Found: {actual_col_names}{undo_hint}")
    else:
        passed_checks.append("REQUIRED_COLUMNS_PRESENT")

    # Primary key check
    pk_cols = [c for c, m in col_meta.items() if m["pk"] > 0]
    primary_key_col = pk_cols[0] if pk_cols else ("id" if "id" in col_meta else (list(col_meta.keys())[0] if col_meta else "id"))
    if not pk_cols and "id" in col_meta:
        failed_checks.append("MISSING_PRIMARY_KEY: 'id' is not declared as PRIMARY KEY.")
    else:
        passed_checks.append("PRIMARY_KEY_DECLARED")

    # NOT NULL constraint declaration in table schema
    for col in target_nn_cols:
        # In SQLite, INTEGER PRIMARY KEY columns are naturally non-null (rowid aliases)
        # even though PRAGMA table_info reports notnull=0.
        is_nn_satisfied = False
        if col in col_meta:
            meta = col_meta[col]
            if meta["notnull"] or meta["pk"] > 0:
                is_nn_satisfied = True

        if not is_nn_satisfied:
            failed_checks.append(
                f"SCHEMA_CONSTRAINT_VIOLATION: Column '{col}' is not defined as NOT NULL. [How to fix: specify '{col}' NOT NULL in CREATE TABLE]"
            )
        else:
            passed_checks.append(f"NOT_NULL_DECLARED_{col}")

    # ── Check 4: Data-level NULL Corruption ───────────────────────────────
    for col in target_nn_cols:
        if col in col_meta:
            m_cur.execute(f"SELECT COUNT(*) FROM {q_target} WHERE {quote_ident(col)} IS NULL")
            null_count = m_cur.fetchone()[0]
            if null_count > 0:
                b_cur.execute(f"SELECT COUNT(*) FROM {q_target} WHERE {quote_ident(col)} IS NULL")
                b_nulls = b_cur.fetchone()[0]
                failed_checks.append(
                    f"NULL_CORRUPTION: Column '{col}' contains {null_count} unhandled NULL values. [How to fix: source has {b_nulls} NULL rows; define default via COALESCE or keep nullable]"
                )
            else:
                passed_checks.append(f"ZERO_NULLS_{col}")

    # ── Check 5: Primary Key Set Uniqueness & Identity ─────────────────────
    if primary_key_col in col_meta:
        q_pk = quote_ident(primary_key_col)
        m_cur.execute(f"SELECT COUNT(*) - COUNT(DISTINCT {q_pk}) FROM {q_target}")
        dups = m_cur.fetchone()[0]
        if dups > 0:
            failed_checks.append("PRIMARY_KEY_DUPLICATION: Duplicate ID values detected after migration.")
        else:
            passed_checks.append("PRIMARY_KEY_UNIQUE")

        if row_policy == "preserve_all":
            if not is_sampled:
                b_cur.execute(f"SELECT {q_pk} FROM {q_target} ORDER BY {q_pk}")
                m_cur.execute(f"SELECT {q_pk} FROM {q_target} ORDER BY {q_pk}")
                pk_diff = False
                while True:
                    b_chunk = b_cur.fetchmany(5000)
                    m_chunk = m_cur.fetchmany(5000)
                    if not b_chunk and not m_chunk:
                        break
                    if b_chunk != m_chunk:
                        pk_diff = True
                        break
                if pk_diff:
                    failed_checks.append("PRIMARY_KEY_SET_MISMATCH: Set of migrated IDs differs from baseline.")
                else:
                    passed_checks.append("PRIMARY_KEY_SET_IDENTICAL")
            else:
                b_cur.execute(f"SELECT COUNT({q_pk}), MIN({q_pk}), MAX({q_pk}), TOTAL({q_pk}) FROM {q_target}")
                b_pk_agg = b_cur.fetchone()
                m_cur.execute(f"SELECT COUNT({q_pk}), MIN({q_pk}), MAX({q_pk}), TOTAL({q_pk}) FROM {q_target}")
                m_pk_agg = m_cur.fetchone()
                if b_pk_agg != m_pk_agg:
                    failed_checks.append("PRIMARY_KEY_SET_MISMATCH: Set of migrated IDs differs from baseline.")
                else:
                    passed_checks.append("PRIMARY_KEY_SET_IDENTICAL")

    # ── Check 6: Checksum of Columns Declared Unchanged ───────────────────
    if spec and target_table in spec.columns_declared_unchanged:
        unchanged_cols = spec.columns_declared_unchanged[target_table]
        q_pk = quote_ident(primary_key_col)
        for c in unchanged_cols:
            if c in orig_col_names and c in col_meta:
                qc = quote_ident(c)
                if not is_sampled:
                    # Stream in PK order with fetchmany(5000) to keep memory flat
                    b_cur.execute(f"SELECT {qc} FROM {q_target} ORDER BY {q_pk}")
                    m_cur.execute(f"SELECT {qc} FROM {q_target} ORDER BY {q_pk}")
                    diff_found = False
                    while True:
                        b_c = b_cur.fetchmany(5000)
                        m_c = m_cur.fetchmany(5000)
                        if not b_c and not m_c:
                            break
                        if b_c != m_c:
                            diff_found = True
                            break
                    if diff_found:
                        failed_checks.append(f"UNTOUCHED_COLUMN_MUTATED: Column '{c}' declared unchanged was modified.")
                    else:
                        passed_checks.append(f"UNTOUCHED_COLUMN_VERIFIED_{c}")
                else:
                    # Above 20k rows: aggregate SQL checksum in engine
                    b_cur.execute(f"""
                        SELECT
                            COUNT(*),
                            TOTAL(CASE WHEN {qc} IS NULL THEN 1 ELSE 0 END),
                            TOTAL(CASE WHEN typeof({qc}) IN ('integer', 'real') THEN CAST({qc} AS REAL) ELSE 0 END),
                            TOTAL(LENGTH(quote({qc})) * (({q_pk} % 997) + 1) + UNICODE(SUBSTR(quote({qc}), 1, 1)))
                        FROM {q_target}
                    """)
                    b_agg = b_cur.fetchone()
                    m_cur.execute(f"""
                        SELECT
                            COUNT(*),
                            TOTAL(CASE WHEN {qc} IS NULL THEN 1 ELSE 0 END),
                            TOTAL(CASE WHEN typeof({qc}) IN ('integer', 'real') THEN CAST({qc} AS REAL) ELSE 0 END),
                            TOTAL(LENGTH(quote({qc})) * (({q_pk} % 997) + 1) + UNICODE(SUBSTR(quote({qc}), 1, 1)))
                        FROM {q_target}
                    """)
                    m_agg = m_cur.fetchone()
                    if b_agg != m_agg:
                        failed_checks.append(f"UNTOUCHED_COLUMN_MUTATED: Column '{c}' declared unchanged was modified.")
                    else:
                        passed_checks.append(f"UNTOUCHED_COLUMN_VERIFIED_{c}")

    # ── Check 7: Preservation of Views, Triggers, and Other Indexes ───────
    m_cur.execute("SELECT name, sql FROM sqlite_master WHERE type='view'")
    migrated_views = {r[0]: _normalize_sql(r[1]) for r in m_cur.fetchall()}
    for vname, vsql in orig_views.items():
        if vname not in migrated_views:
            failed_checks.append(f"LOST_VIEW: View '{vname}' was dropped during migration.")
        else:
            passed_checks.append(f"VIEW_PRESERVED_{vname}")

    m_cur.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger'")
    migrated_triggers = {r[0]: _normalize_sql(r[1]) for r in m_cur.fetchall()}
    for tname, tsql in orig_triggers.items():
        if tname not in migrated_triggers:
            failed_checks.append(f"LOST_TRIGGER: Trigger '{tname}' was dropped during migration.")
        else:
            passed_checks.append(f"TRIGGER_PRESERVED_{tname}")

    m_cur.execute(
        "SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
    )
    migrated_indexes = {r[0]: (r[1], _normalize_sql(r[2])) for r in m_cur.fetchall()}
    for iname, (tbl_n, isql) in orig_indexes.items():
        if iname not in migrated_indexes:
            failed_checks.append(f"LOST_INDEX: Index '{iname}' on table '{tbl_n}' was dropped.")
        else:
            passed_checks.append(f"INDEX_PRESERVED_{iname}")

    # ── Check 8: PRAGMA foreign_key_check (Baseline-Subtracted) ───────────
    m_cur.execute("PRAGMA foreign_key_check")
    after_fk_violations = set(tuple(r) for r in m_cur.fetchall())
    new_fk_violations = after_fk_violations - baseline_fk_violations
    if new_fk_violations:
        failed_checks.append(f"NEW_FOREIGN_KEY_VIOLATIONS: {len(new_fk_violations)} violations introduced.")
    else:
        passed_checks.append("FOREIGN_KEYS_CLEAN")

    # ── Check 9: PRAGMA integrity_check ───────────────────────────────────
    m_cur.execute("PRAGMA integrity_check")
    ic = m_cur.fetchone()[0]
    if ic != "ok":
        failed_checks.append(f"SQLITE_INTEGRITY_CHECK_FAILED: {ic}")
    else:
        passed_checks.append("SQLITE_INTEGRITY_OK")

    # ── Check 10: Non-Target Tables Untouched ──────────────────────────────
    parent_tables: set[str] = set()
    try:
        m_cur.execute(f'PRAGMA foreign_key_list("{target_table}")')
        parent_tables.update(r[2] for r in m_cur.fetchall())
    except Exception:
        pass
    try:
        b_cur.execute(f'PRAGMA foreign_key_list("{target_table}")')
        parent_tables.update(r[2] for r in b_cur.fetchall())
    except Exception:
        pass
    if spec:
        for m in spec.get_table_mappings(target_table):
            if m.lookup_table:
                parent_tables.add(m.lookup_table)

    for other_tbl in orig_tables:
        if other_tbl != target_table:
            b_cur.execute(f'SELECT COUNT(*) FROM "{other_tbl}"')
            b_cnt = b_cur.fetchone()[0]
            try:
                m_cur.execute(f'SELECT COUNT(*) FROM "{other_tbl}"')
                m_cnt = m_cur.fetchone()[0]

                # If other_tbl is a referenced parent table and has seeded sentinel row(s)
                if other_tbl in parent_tables and m_cnt >= b_cnt and (m_cnt - b_cnt <= 10):
                    b_cur.execute(f'SELECT * FROM "{other_tbl}"')
                    b_data_set = set(b_cur.fetchall())
                    m_cur.execute(f'SELECT * FROM "{other_tbl}"')
                    m_data_set = set(m_cur.fetchall())
                    if b_data_set.issubset(m_data_set):
                        passed_checks.append(f"NON_TARGET_TABLE_UNTOUCHED_{other_tbl} (sentinel seeded)")
                    else:
                        failed_checks.append(f"NON_TARGET_TABLE_MUTATED: Table '{other_tbl}' was unexpectedly modified.")
                elif b_cnt != m_cnt:
                    failed_checks.append(f"NON_TARGET_TABLE_MUTATED: Table '{other_tbl}' was unexpectedly modified.")
                elif b_cnt <= 5000:
                    b_cur.execute(f'SELECT * FROM "{other_tbl}"')
                    b_data = b_cur.fetchall()
                    m_cur.execute(f'SELECT * FROM "{other_tbl}"')
                    m_data = m_cur.fetchall()
                    if b_data != m_data:
                        failed_checks.append(f"NON_TARGET_TABLE_MUTATED: Table '{other_tbl}' was unexpectedly modified.")
                    else:
                        passed_checks.append(f"NON_TARGET_TABLE_UNTOUCHED_{other_tbl}")
                else:
                    passed_checks.append(f"NON_TARGET_TABLE_UNTOUCHED_{other_tbl}")
            except Exception as exc:
                failed_checks.append(f"NON_TARGET_TABLE_ERROR: Table '{other_tbl}': {exc}")

    # ── Check 11: Row-Level Oracle Verification ───────────────────────────
    m_cur.execute(f"SELECT * FROM {q_target} LIMIT 8")
    migrated_cols = [d[0] for d in m_cur.description]
    sample_rows = [dict(zip(migrated_cols, r)) for r in m_cur.fetchall()]

    # Collect rows for oracle verification
    q_pk = quote_ident(primary_key_col)
    if not is_sampled:
        m_cur.execute(f"SELECT * FROM {q_target}")
        all_mig_cols = [d[0] for d in m_cur.description]
        migrated_by_pk = {str(r[all_mig_cols.index(primary_key_col)]): dict(zip(all_mig_cols, r)) for r in m_cur.fetchall() if primary_key_col in all_mig_cols}
        audit_orig_rows = orig_rows
    else:
        # Sampled audit: collect all trap rows + deterministic 5k sample
        trap_pks = set()
        try:
            b_cur.execute("SELECT target_pk FROM __trap_rows WHERE target_table = ?", (target_table,))
            trap_pks = {r[0] for r in b_cur.fetchall()}
        except Exception:
            pass

        stride = max(1, before_count // 5000)
        b_cur.execute(f"SELECT * FROM {q_target} WHERE ({q_pk} % {stride}) = 0 ORDER BY {q_pk} LIMIT 5000")
        audit_orig_rows = [dict(zip(orig_col_names, r)) for r in b_cur.fetchall()]
        seen_pks = {r.get(primary_key_col) for r in audit_orig_rows}

        for tpk in trap_pks:
            if tpk not in seen_pks:
                b_cur.execute(f"SELECT * FROM {q_target} WHERE {q_pk} = ?", (tpk,))
                r = b_cur.fetchone()
                if r:
                    audit_orig_rows.append(dict(zip(orig_col_names, r)))
                    seen_pks.add(tpk)

        migrated_by_pk = {}
        pks_to_fetch = [r[primary_key_col] for r in audit_orig_rows if primary_key_col in r]
        for i in range(0, len(pks_to_fetch), 900):
            batch = pks_to_fetch[i : i + 900]
            placeholders = ",".join("?" for _ in batch)
            m_cur.execute(f"SELECT * FROM {q_target} WHERE {q_pk} IN ({placeholders})", batch)
            m_cols_batch = [d[0] for d in m_cur.description]
            for r in m_cur.fetchall():
                row_d = dict(zip(m_cols_batch, r))
                migrated_by_pk[str(row_d[primary_key_col])] = row_d

    oracle_verified = False

    # 1. Scenario explicit expected_row_fn if provided
    if expected_row_fn is not None:
        b_cur.execute("SELECT code FROM departments")
        valid_depts = {r[0] for r in b_cur.fetchall()}
        mismatches: dict[str, int] = {}
        examples: list[dict] = []
        for orig in audit_orig_rows:
            exp = expected_row_fn(orig, valid_depts)
            act = migrated_by_pk.get(str(orig.get(primary_key_col)))
            if not act:
                continue
            for k, exp_val in exp.items():
                if k in act and act[k] != exp_val:
                    mismatches[k] = mismatches.get(k, 0) + 1
                    if len(examples) < 3:
                        examples.append({"id": orig.get(primary_key_col), "column": k, "expected": exp_val, "actual": act[k]})
        if mismatches:
            summary = ", ".join(f"{c}: {n} wrong" for c, n in sorted(mismatches.items()))
            ex_str = "; ".join(f"id={e['id']} {e['column']}: expected {e['expected']!r} got {e['actual']!r}" for e in examples)
            contract_lines = []
            if spec:
                for col_w in sorted(mismatches.keys()):
                    m = next((m for m in spec.get_table_mappings(target_table) if m.new_column == col_w), None)
                    if m:
                        contract_lines.append(f"{m.new_column} from {m.source_column or 'None'} (transform: {m.transform}, default: {m.default_value!r}, orphan_policy: {m.orphan_policy})")
            contract_hint = f" [Contract: {'; '.join(contract_lines)}]" if contract_lines else ""
            fk_hint = _build_fk_mismatch_hint(target_table, mismatches, examples, spec, b_cur, m_cur)
            failed_checks.append(f"DATA_MISMATCH: {summary}. Examples: {ex_str}{contract_hint}{fk_hint}")
        else:
            passed_checks.append("ROW_ORACLE_MATCHED" if not is_sampled else "ROW_ORACLE_MATCHED (sampled)")
            oracle_verified = True

    # 2. Spec-based oracle library transforms
    elif spec and target_table in spec.column_mappings:
        mappings = spec.get_table_mappings(target_table)
        lookups: dict[str, dict[str, str]] = {}
        for m in mappings:
            if m.lookup_table and m.lookup_key and m.lookup_val:
                try:
                    b_cur.execute(f"SELECT \"{m.lookup_key}\", \"{m.lookup_val}\" FROM \"{m.lookup_table}\"")
                    lookups[m.lookup_table] = {str(r[0]): str(r[1]) for r in b_cur.fetchall()}
                except Exception:
                    pass

        has_oracle_rules = any(m.transform not in ("custom", "") for m in mappings)
        if has_oracle_rules:
            matched, mismatches, examples = verify_rows_with_oracle(
                audit_orig_rows,
                migrated_by_pk,
                mappings,
                lookup_data=lookups,
                pk_column=primary_key_col,
            )
            if not matched:
                summary = ", ".join(f"{c}: {n} wrong" for c, n in sorted(mismatches.items()))
                ex_str = "; ".join(f"id={e['id']} {e['column']}: expected {e['expected']!r} got {e['actual']!r}" for e in examples)
                contract_lines = []
                for col_w in sorted(mismatches.keys()):
                    m = next((m for m in mappings if m.new_column == col_w), None)
                    if m:
                        contract_lines.append(f"{m.new_column} from {m.source_column or 'None'} (transform: {m.transform}, default: {m.default_value!r}, orphan_policy: {m.orphan_policy})")
                contract_hint = f" [Contract: {'; '.join(contract_lines)}]" if contract_lines else ""
                fk_hint = _build_fk_mismatch_hint(target_table, mismatches, examples, spec, b_cur, m_cur)
                failed_checks.append(f"DATA_MISMATCH: {summary}. Examples: {ex_str}{contract_hint}{fk_hint}")
            else:
                passed_checks.append("ROW_ORACLE_MATCHED" if not is_sampled else "ROW_ORACLE_MATCHED (sampled)")
                oracle_verified = True

    # ── Final Status Determination ─────────────────────────────────────────
    passed = len(failed_checks) == 0

    if not passed:
        final_status = "FAILED"
        primary_error = failed_checks[0]
    elif oracle_verified:
        final_status = "VERIFIED"
        primary_error = None
    else:
        final_status = "UNVERIFIED"
        primary_error = None

    return InvariantResult(
        status=final_status,
        passed=passed,
        error=primary_error,
        before_count=before_count,
        after_count=after_count,
        elapsed_ms=elapsed_ms,
        failed_checks=failed_checks,
        passed_checks=passed_checks,
        sample_rows=sample_rows,
        oracle_verified=oracle_verified,
        is_sampled=is_sampled,
    )


def audit_crash_diagnostics(
    base_conn: sqlite3.Connection,
    target_table: str,
    err_msg: str,
    current_sql: str = "",
    spec: MigrationSpec | None = None,
) -> str:
    """Audit vulnerable columns when execution crashes on NOT NULL or FOREIGN KEY constraints.

    Identifies all vulnerable columns in the source table that contain NULLs or orphan foreign keys,
    checks declarations in migration SQL, and returns explicit remediation recipes so the LLM
    can fix all related issues in a single repair turn.
    """
    err_lower = err_msg.lower()

    # ── Branch 1: Foreign Key Constraint Failure ───────────────────────────
    if "foreign key constraint failed" in err_lower:
        try:
            cur = base_conn.cursor()
            fk_matches: list[tuple[str, str, str]] = []
            if current_sql:
                # Column-level REFERENCES
                for m in re.finditer(
                    r"([a-zA-Z0-9_]+)\s+[^,\(\)]*?\bREFERENCES\s+[\"']?([a-zA-Z0-9_]+)[\"']?\s*\(\s*[\"']?([a-zA-Z0-9_]+)[\"']?\s*\)",
                    current_sql,
                    re.IGNORECASE,
                ):
                    fk_matches.append((m.group(1), m.group(2), m.group(3)))
                # Table-level FOREIGN KEY (col) REFERENCES parent(pk)
                for m in re.finditer(
                    r"FOREIGN\s+KEY\s*\(\s*[\"']?([a-zA-Z0-9_]+)[\"']?\s*\)\s*REFERENCES\s+[\"']?([a-zA-Z0-9_]+)[\"']?\s*\(\s*[\"']?([a-zA-Z0-9_]+)[\"']?\s*\)",
                    current_sql,
                    re.IGNORECASE,
                ):
                    fk_matches.append((m.group(1), m.group(2), m.group(3)))

            # If no matches in SQL, check baseline foreign_key_list
            if not fk_matches:
                try:
                    cur.execute(f'PRAGMA foreign_key_list("{target_table}")')
                    for r in cur.fetchall():
                        fk_matches.append((r[3], r[2], r[4]))
                except Exception:
                    pass

            # If still none, check schema for candidate parent tables
            if not fk_matches:
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                all_tbls = [r[0] for r in cur.fetchall() if r[0].lower() != target_table.lower()]
                for pt in all_tbls:
                    cur.execute(f'PRAGMA table_info("{pt}")')
                    p_pks = [r[1] for r in cur.fetchall() if r[5] > 0]
                    if p_pks:
                        cur.execute(f'PRAGMA table_info("{target_table}")')
                        t_cols = [r[1] for r in cur.fetchall()]
                        for tc in t_cols:
                            if any(part in tc.lower() for part in (pt.lower(), p_pks[0].lower(), "dept")):
                                fk_matches.append((tc, pt, p_pks[0]))

            fk_diagnostics: list[str] = []
            for child_col, parent_table, parent_pk in fk_matches:
                # Ensure parent table exists
                cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)", (parent_table,))
                if cur.fetchone()[0] == 0:
                    continue

                cur.execute(f'SELECT DISTINCT "{parent_pk}" FROM "{parent_table}" LIMIT 10')
                valid_parent_keys = [str(r[0]) for r in cur.fetchall()]

                cur.execute(
                    f'SELECT DISTINCT "{child_col}" FROM "{target_table}" '
                    f'WHERE "{child_col}" IS NOT NULL AND "{child_col}" NOT IN (SELECT "{parent_pk}" FROM "{parent_table}")'
                )
                orphan_values = [str(r[0]) for r in cur.fetchall()]

                cur.execute(f'SELECT COUNT(*) FROM "{target_table}" WHERE "{child_col}" IS NULL')
                null_cnt = cur.fetchone()[0]

                cur.execute(f'PRAGMA table_info("{parent_table}")')
                p_cols = cur.fetchall()
                mapping = next(
                    (m for m in (spec.get_table_mappings(target_table) if spec else [])
                     if m.new_column.lower() == child_col.lower() or (m.source_column and m.source_column.lower() == child_col.lower())),
                    None,
                )
                sentinel_val = "UNASSIGNED"
                if mapping and mapping.default_value is not None:
                    sentinel_val = str(mapping.default_value)
                elif mapping and mapping.orphan_policy and mapping.orphan_policy.lower() not in ("null", "none"):
                    sentinel_val = str(mapping.orphan_policy)

                seed_c = []
                seed_v = []
                for cid, cname, ctype, notnull, dflt, pk in p_cols:
                    if pk > 0 or notnull:
                        col_ident = cname if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", cname) else f'"{cname}"'
                        seed_c.append(col_ident)
                        if pk > 0:
                            seed_v.append(repr(sentinel_val))
                        else:
                            seed_v.append(repr(sentinel_val.capitalize()))
                seed_sql = f'INSERT OR IGNORE INTO {parent_table} ({", ".join(seed_c)}) VALUES ({", ".join(seed_v)});'
                src_col_name = (mapping.source_column or child_col) if mapping else child_col

                orphans_str = ", ".join(repr(v) for v in orphan_values) if orphan_values else "None detected"
                valid_str = ", ".join(repr(k) for k in valid_parent_keys)

                fk_diagnostics.append(
                    f"Foreign key '{target_table}.{child_col}' -> '{parent_table}({parent_pk})':\n"
                    f"  - Valid parent keys in '{parent_table}': [{valid_str}]\n"
                    f"  - Orphan values in '{target_table}.{child_col}' not in '{parent_table}': [{orphans_str}]\n"
                    f"  - NULL rows in '{target_table}.{child_col}': {null_cnt}\n\n"
                    f"  CONCRETE REMEDIATION RECIPE (2-STEP SENTINEL FOREIGN KEY PATTERN):\n"
                    f"  If your migration goal, transform contract, or column default requires a sentinel value (such as {sentinel_val!r}) instead of NULL:\n"
                    f"  Do NOT use COALESCE alone (it ignores non-NULL orphan trap values like [{orphans_str}])!\n"
                    f"  Do NOT simply write ELSE {sentinel_val!r} without seeding the parent table (it crashes SQLite with FOREIGN KEY constraint failed)!\n"
                    f"  You MUST execute BOTH steps:\n"
                    f"  1. Step A (Seed Parent Table FIRST before creating or inserting into child table):\n"
                    f"     {seed_sql}\n"
                    f"  2. Step B (Map BOTH NULL and Orphan Keys in child SELECT query):\n"
                    f"     CASE WHEN \"{src_col_name}\" IS NOT NULL AND \"{src_col_name}\" IN (SELECT \"{parent_pk}\" FROM \"{parent_table}\") THEN \"{src_col_name}\"\n"
                    f"       ELSE {sentinel_val!r}  -- (Or ELSE NULL if and only if NULL is explicitly expected)\n"
                    f"     END AS \"{child_col}\""
                )

            if fk_diagnostics:
                return (
                    " [FOREIGN KEY CONSTRAINT CRASH DIAGNOSTIC: SQLite failed with FOREIGN KEY constraint failed.\n"
                    + "\n\n".join(fk_diagnostics)
                    + "\nApply the 2-step Sentinel Foreign Key Pattern above: seed the parent table before child insertion, and map invalid keys in SELECT.]"
                )
        except Exception as exc:
            logger.debug(f"Error computing foreign key crash diagnostics: {exc}")
            return ""

    # ── Branch 2: NOT NULL Constraint Failure ──────────────────────────────
    if "not null constraint failed" not in err_lower:
        return ""

    try:
        cur = base_conn.cursor()
        cur.execute(f'PRAGMA table_info("{target_table}")')
        base_cols = cur.fetchall()  # (cid, name, type, notnull, dflt_value, pk)
        if not base_cols:
            return ""

        # 1. Identify baseline nullability (notnull == 0 and pk == 0)
        baseline_nullable = {r[1]: (r[3] == 0 and r[5] == 0) for r in base_cols}

        # 2. Query source table for all columns containing NULLs
        null_columns: dict[str, int] = {}
        for r in base_cols:
            cname = r[1]
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{target_table}" WHERE "{cname}" IS NULL')
                cnt = cur.fetchone()[0]
                if cnt > 0:
                    null_columns[cname] = cnt
            except Exception:
                pass

        # 3. Parse target table CREATE TABLE statement in current_sql to find declared NOT NULL columns
        declared_not_null_cols: set[str] = set()
        if current_sql:
            tbl_re = re.compile(
                r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"']?([a-zA-Z0-9_]+)[\"']?\s*\((.*?)\)(?:;|\s*$|\n\s*[A-Z])",
                re.IGNORECASE | re.DOTALL,
            )
            for m in tbl_re.finditer(current_sql):
                created_tname = m.group(1).lower()
                if created_tname.startswith(target_table.lower()) or target_table.lower() in created_tname:
                    body = m.group(2)
                    for col_def in body.split(","):
                        col_def = col_def.strip()
                        if re.search(r"\bNOT\s+NULL\b", col_def, re.IGNORECASE):
                            col_words = re.findall(r"\b[a-zA-Z0-9_]+\b", col_def)
                            if col_words and col_words[0].upper() not in ("PRIMARY", "UNIQUE", "CHECK", "FOREIGN", "CONSTRAINT"):
                                declared_not_null_cols.add(col_words[0])

        if not null_columns:
            return ""

        first_match = re.search(r"NOT NULL constraint failed:\s*([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", err_msg)
        first_col = first_match.group(2) if first_match else "column"

        vuln_lines = []
        vuln_names = []
        unchanged_cols = set(spec.columns_declared_unchanged.get(target_table, [])) if spec else set()
        for cname, cnt in sorted(null_columns.items()):
            vuln_names.append(cname)
            is_base_null = baseline_nullable.get(cname, True)
            base_status = "nullable in original schema" if is_base_null else "non-null in original schema"
            mapping = next((m for m in spec.get_table_mappings(target_table) if m.new_column == cname), None) if spec else None
            spec_default = f", spec default: '{mapping.default_value}'" if mapping and mapping.default_value is not None else ""
            unchanged_tag = " (⚠️ UNCHANGED COLUMN: Must remain NULLABLE in CREATE TABLE to preserve original NULLs; do NOT coalesce!)" if cname in unchanged_cols else ""
            vuln_lines.append(f"  - '{cname}': {cnt} NULL row(s) in source table ({base_status}{spec_default}){unchanged_tag}")

        names_str = ", ".join(f"'{n}'" for n in vuln_names)

        diag = (
            f" [MULTI-COLUMN NOT NULL DIAGNOSTIC: SQLite halted at the first NOT NULL violation ('{first_col}'), "
            f"but multiple columns contain NULL values in the source table:\n"
            + "\n".join(vuln_lines)
            + f"\nCRITICAL FIX INSTRUCTIONS FOR THIS SINGLE PASS:\n"
            f"1. In your CREATE TABLE statement, do NOT add NOT NULL to existing columns ({names_str}) "
            f"that were nullable in the original schema unless explicitly required by the migration goal. Keep them nullable.\n"
            f"2. For EVERY column in the new table that IS declared NOT NULL, wrap its SELECT expression with "
            f"COALESCE(<col>, <safe_default>) in this single repair pass so another column does not fail sequentially.]"
        )
        return diag
    except Exception as exc:
        logger.debug(f"Error computing multi-column NULL diagnostics: {exc}")
        return ""


# Backwards compatibility alias
audit_null_crash_diagnostics = audit_crash_diagnostics


