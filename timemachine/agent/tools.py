"""OpenAI-style tool definitions and execution registry for agent."""

import json
import sqlite3
from typing import Any

from timemachine.adapters.base import DbAdapter
from timemachine.agent.llm import has_ddl_or_dml
from timemachine.agent.prompts import format_schema_summary
from timemachine.traps.profiler import mask_value, profile_table
from timemachine.verify.invariants import evaluate_migration_invariants
from timemachine.verify.spec import MigrationSpec

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Inspect database tables, columns, data types, primary keys, and foreign keys.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "profile_table",
            "description": "Get statistical summary and masked sample rows of a specific table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Name of the table to profile"}
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_violating_rows",
            "description": "Find sample rows matching a SQL condition with PII masked (limit <= 5).",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Table to query"},
                    "predicate": {"type": "string", "description": "WHERE clause predicate (e.g. 'dept_code IS NULL')"},
                    "limit": {"type": "integer", "description": "Maximum sample rows (default 5)", "default": 5},
                },
                "required": ["table_name", "predicate"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_on_clone",
            "description": "Execute up_sql and optional down_sql on a disposable RAM clone and audit invariants.",
            "parameters": {
                "type": "object",
                "properties": {
                    "up_sql": {"type": "string", "description": "SQLite schema migration script"},
                    "down_sql": {"type": "string", "description": "Rollback script (optional)", "default": ""},
                },
                "required": ["up_sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain",
            "description": "Run EXPLAIN QUERY PLAN on the disposable sandbox clone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL statement to explain"}
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit final verified up_sql and down_sql scripts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "up_sql": {"type": "string", "description": "Final forward migration SQL"},
                    "down_sql": {"type": "string", "description": "Final rollback SQL script to restore baseline state without data loss", "default": ""},
                    "notes": {"type": "string", "description": "Explanatory notes and trade-offs", "default": ""},
                },
                "required": ["up_sql"],
            },
        },
    },
]


class ToolExecutor:
    """Executes agent tool calls against the database adapter and disposable sandbox."""

    def __init__(
        self,
        adapter: DbAdapter,
        spec: MigrationSpec | None = None,
        target_table: str = "employees",
        base_snapshot: Any | None = None,
    ):
        self.adapter = adapter
        self.spec = spec
        self.target_table = target_table
        self.base_snapshot = base_snapshot
        self._schema_call_count = 0
        self._cached_schema: Any | None = None
        self._cached_schema_summary: str = ""
        self._profile_cache: dict[str, dict[str, Any]] = {}

    def create_clone(self) -> Any:
        """Create disposable sandbox clone using fast in-memory backup if snapshot is available."""
        from timemachine.verify.sandbox import apply_sandbox

        timeout = getattr(self.adapter.settings, "sandbox_timeout_s", 2.0) if hasattr(self.adapter, "settings") else 2.0
        if self.base_snapshot is not None:
            c = sqlite3.connect(":memory:", check_same_thread=False)
            self.base_snapshot.backup(c)
            apply_sandbox(c, deadline_s=timeout)
            return c
        c = self.adapter.clone()
        apply_sandbox(c, deadline_s=timeout)
        return c

    def execute(self, name: str, arguments_json: str) -> dict[str, Any]:
        """Dispatch tool call by name."""
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except Exception as exc:
            return {"error": f"Invalid arguments JSON: {exc}"}

        try:
            if name == "get_schema":
                if self._cached_schema is None:
                    self._cached_schema = self.adapter.introspect()
                    self._cached_schema_summary = format_schema_summary(self._cached_schema)
                if self._schema_call_count > 0:
                    return {
                        "note": "Schema already provided in previous round.",
                        "schema_summary": self._cached_schema_summary,
                    }
                self._schema_call_count += 1
                return {"schema_summary": self._cached_schema_summary}

            if name == "profile_table":
                tname = args.get("table_name", self.target_table)
                if tname in self._profile_cache:
                    return {
                        "note": f"Profile for table '{tname}' already provided.",
                        "table": tname,
                        "profiles": self._profile_cache[tname],
                    }
                if self._cached_schema is None:
                    self._cached_schema = self.adapter.introspect()
                    self._cached_schema_summary = format_schema_summary(self._cached_schema)
                tbl = self._cached_schema.get_table(tname)
                if not tbl:
                    return {"error": f"Table '{tname}' not found in database."}
                clone = self.create_clone()
                try:
                    prof = profile_table(clone, tbl)
                    prof_data = {k: v.model_dump() for k, v in prof.items()}
                    self._profile_cache[tname] = prof_data
                    return {"table": tname, "profiles": prof_data}
                finally:
                    clone.close()

            if name == "find_violating_rows":
                tname = args.get("table_name", self.target_table)
                pred = args.get("predicate", "1=1")
                try:
                    limit_raw = args.get("limit", 5)
                    limit_val = int(limit_raw)
                except (ValueError, TypeError):
                    limit_val = 5
                limit = max(1, min(limit_val, 5))
                clone = self.create_clone()
                try:
                    cur = clone.cursor()
                    cur.execute(f"SELECT * FROM \"{tname}\" WHERE {pred} LIMIT {limit}")
                    cols = [d[0] for d in cur.description]
                    rows = cur.fetchall()
                    masked_rows = [
                        {cols[i]: mask_value(r[i]) for i in range(len(cols))}
                        for r in rows
                    ]
                    return {"count": len(masked_rows), "sample_rows": masked_rows}
                finally:
                    clone.close()

            if name == "run_on_clone":
                up_sql = args.get("up_sql", "")
                if not up_sql or not up_sql.strip():
                    return {"error": "up_sql must not be empty."}

                import re

                is_read_query = bool(re.match(r"^\s*(SELECT|PRAGMA|EXPLAIN)\b", up_sql, re.IGNORECASE))
                if is_read_query:
                    clone = self.create_clone()
                    try:
                        cur = clone.cursor()
                        cur.execute(up_sql)
                        cols = [d[0] for d in cur.description] if cur.description else []
                        rows = cur.fetchall()
                        limit = 10
                        sample_rows = [
                            {cols[i]: mask_value(r[i]) for i in range(len(cols))}
                            for r in rows[:limit]
                        ]
                        return {
                            "status": "query_result",
                            "note": "Executed read query on sandbox clone.",
                            "columns": cols,
                            "count": len(rows),
                            "sample_rows": sample_rows,
                        }
                    except Exception as q_err:
                        return {"error": f"Query execution error: {q_err}"}
                    finally:
                        clone.close()

                base_clone = self.create_clone()
                test_clone = self.create_clone()
                try:
                    ok, err, elapsed = self.adapter.run_in_sandbox(test_clone, up_sql, fk_enabled=True)
                    if not ok:
                        return {
                            "status": "FAILED",
                            "passed": False,
                            "error": err,
                            "elapsed_ms": elapsed,
                            "failed_checks": [err],
                        }

                    inv = evaluate_migration_invariants(
                        base_clone,
                        test_clone,
                        target_table=self.target_table,
                        spec=self.spec,
                        elapsed_ms=elapsed,
                    )
                    try:
                        from timemachine.traps.generator import audit_traps
                        traps = audit_traps(test_clone, self.target_table)
                        inv.trap_audit = traps
                        failing_traps = [t for t in traps if not t.get("survived")]
                        if failing_traps:
                            inv.status = "FAILED"
                            inv.passed = False
                            for ft in failing_traps:
                                inv.failed_checks.append(
                                    f"ADVERSARIAL_TRAP_FAILED: Trap '{ft.get('trap_type')}' failed ({ft.get('description')})"
                                )
                    except Exception:
                        pass

                    res_dict = inv.model_dump()
                    down_sql = args.get("down_sql", "").strip()
                    if down_sql and has_ddl_or_dml(down_sql):
                        from timemachine.verify.rebuild import verify_rollback_roundtrip
                        rb_clone = self.create_clone()
                        try:
                            rb_res = verify_rollback_roundtrip(
                                rb_clone,
                                up_sql,
                                down_sql,
                                target_table=self.target_table,
                                base_timeout_s=getattr(self.adapter.settings, "sandbox_timeout_s", 2.0) if hasattr(self.adapter, "settings") else 2.0,
                            )
                            res_dict["rollback_verified"] = rb_res.rollback_verified
                            res_dict["rollback_classification"] = rb_res.classification
                            if not rb_res.rollback_verified:
                                res_dict["rollback_error"] = rb_res.error
                                res_dict["rollback_lossy_columns"] = rb_res.lossy_columns
                        finally:
                            rb_clone.close()

                    return res_dict
                finally:
                    base_clone.close()
                    test_clone.close()

            if name == "explain":
                sql = args.get("sql", "")
                clone = self.create_clone()
                try:
                    cur = clone.cursor()
                    cur.execute(f"EXPLAIN QUERY PLAN {sql}")
                    plans = [str(r) for r in cur.fetchall()]
                    return {"query_plan": plans}
                finally:
                    clone.close()

            if name == "submit":
                return {
                    "status": "submitted",
                    "up_sql": args.get("up_sql", ""),
                    "down_sql": args.get("down_sql", ""),
                    "notes": args.get("notes", ""),
                }

            return {"error": f"Unknown tool: '{name}'"}
        except Exception as exc:
            return {"error": f"Tool execution failed: {exc}"}
