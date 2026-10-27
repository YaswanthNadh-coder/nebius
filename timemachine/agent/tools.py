"""OpenAI-style tool definitions and execution registry for agent."""

import json
from typing import Any

from timemachine.adapters.base import DbAdapter
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
                    "down_sql": {"type": "string", "description": "Final rollback SQL"},
                    "notes": {"type": "string", "description": "Explanatory notes and trade-offs"},
                },
                "required": ["up_sql", "down_sql"],
            },
        },
    },
]


class ToolExecutor:
    """Executes agent tool calls against the database adapter and disposable sandbox."""

    def __init__(self, adapter: DbAdapter, spec: MigrationSpec | None = None, target_table: str = "employees"):
        self.adapter = adapter
        self.spec = spec
        self.target_table = target_table

    def execute(self, name: str, arguments_json: str) -> dict[str, Any]:
        """Dispatch tool call by name."""
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except Exception:
            args = {}

        if name == "get_schema":
            schema = self.adapter.introspect()
            return {"schema": schema.model_dump()}

        if name == "profile_table":
            tname = args.get("table_name", self.target_table)
            schema = self.adapter.introspect()
            tbl = schema.get_table(tname)
            if not tbl:
                return {"error": f"Table '{tname}' not found in database."}
            clone = self.adapter.clone()
            try:
                prof = profile_table(clone, tbl)
                return {"table": tname, "profiles": {k: v.model_dump() for k, v in prof.items()}}
            finally:
                clone.close()

        if name == "find_violating_rows":
            tname = args.get("table_name", self.target_table)
            pred = args.get("predicate", "1=1")
            limit = min(int(args.get("limit", 5)), 5)
            clone = self.adapter.clone()
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
            except Exception as exc:
                return {"error": str(exc)}
            finally:
                clone.close()

        if name == "run_on_clone":
            up_sql = args.get("up_sql", "")
            base_clone = self.adapter.clone()
            test_clone = self.adapter.clone()
            try:
                ok, err, elapsed = self.adapter.run_in_sandbox(test_clone, up_sql, fk_enabled=True)
                if not ok:
                    return {
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
                return inv.model_dump()
            finally:
                base_clone.close()
                test_clone.close()

        if name == "explain":
            sql = args.get("sql", "")
            clone = self.adapter.clone()
            try:
                cur = clone.cursor()
                cur.execute(f"EXPLAIN QUERY PLAN {sql}")
                plans = [str(r) for r in cur.fetchall()]
                return {"query_plan": plans}
            except Exception as exc:
                return {"error": str(exc)}
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
