"""Command line interface for SQL Migration Dry-Run Time Machine."""

import argparse
import os
import sys
from pathlib import Path

# Configure stdout for safe UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.agent.loop import run_time_machine_agent
from timemachine.config import get_api_key_info, is_placeholder_key, load_env_file
from timemachine.logging import get_logger
from timemachine.report.history import save_run_history
from timemachine.report.writer import (
    export_alembic,
    export_flyway,
    generate_markdown_report,
    wrap_transaction_with_prechecks,
)

logger = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="timemachine",
        description="Autonomous SQL Migration Dry-Run Time Machine CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Run migration dry-run on a database clone")
    run_p.add_argument("--db", type=str, default="", help="Path to SQLite database file (.db or .sqlite)")
    run_p.add_argument("--goal", type=str, required=True, help="Migration goal text or path to a text file")
    run_p.add_argument("--out", type=str, default="migrations", help="Output directory for generated files")
    run_p.add_argument("--table", type=str, default="", help="Primary target table name (auto-detected if omitted)")
    run_p.add_argument("--max-steps", type=int, default=8, help="Maximum agent planning/repair steps (default 8)")
    run_p.add_argument("--key", type=str, default="", help="Nebius Token Factory API key override")
    run_p.add_argument("--format", choices=["plain", "flyway", "alembic"], default="plain", help="Export format")

    return parser


def main(args: list[str] | None = None) -> int:
    """CLI execution entrypoint. Returns exit code: 0=VERIFIED, 1=FAILED, 2=UNVERIFIED."""
    load_env_file()
    parser = build_parser()
    parsed = parser.parse_args(args)

    if parsed.command != "run":
        parser.print_help()
        return 1

    # Resolve goal
    goal = parsed.goal
    if os.path.isfile(goal):
        with open(goal, encoding="utf-8-sig") as f:
            goal = f.read().strip()

    # Resolve API Key
    clean_key, key_src, masked_key = get_api_key_info(parsed.key)
    if is_placeholder_key(clean_key):
        print(f"[ERROR] No valid Nebius API key found. Checked {key_src}.", file=sys.stderr)
        return 1

    # Connect to DB
    if parsed.db and os.path.exists(parsed.db):
        adapter = SQLiteAdapter.from_file(parsed.db)
        print(f"[INFO] Connected to source database: {parsed.db} (read-only)")
    else:
        # Fallback to in-memory demo scenario 1
        from timemachine.ui.app_view import create_builtin_scenario_adapter
        adapter = create_builtin_scenario_adapter(1)
        print("[INFO] No database specified; loaded built-in Employee Normalization demo.")

    schema = adapter.introspect()
    target_table = parsed.table
    if not target_table:
        # Pick first table in schema
        t_names = list(schema.tables.keys())
        target_table = "employees" if "employees" in t_names else (t_names[0] if t_names else "main")

    print(f"[INFO] Target Table: {target_table}")
    print(f"[INFO] Migration Goal: {goal}")
    print("[INFO] Starting autonomous dry-run verification...")

    try:
        result = run_time_machine_agent(
            adapter=adapter,
            goal=goal,
            target_table=target_table,
            api_key_override=clean_key,
            max_steps=parsed.max_steps,
            progress_callback=lambda msg, data: print(f"  -> {msg}"),
        )
    except Exception as exc:
        print(f"[ERROR] Agent run failed with exception: {exc}", file=sys.stderr)
        return 1

    # Save to history
    save_run_history(result, target_table, goal)

    # Write outputs
    out_dir = Path(parsed.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapped_up = wrap_transaction_with_prechecks(result.up_sql, target_table, result.spec)

    if parsed.format == "plain":
        (out_dir / "up.sql").write_text(wrapped_up, encoding="utf-8")
        (out_dir / "down.sql").write_text(result.down_sql, encoding="utf-8")
    elif parsed.format == "flyway":
        files = export_flyway(wrapped_up, result.down_sql)
        for fname, content in files.items():
            (out_dir / fname).write_text(content, encoding="utf-8")
    elif parsed.format == "alembic":
        alembic_code = export_alembic(result.up_sql, result.down_sql)
        (out_dir / "alembic_migration.py").write_text(alembic_code, encoding="utf-8")

    report_md = generate_markdown_report(result, target_table, goal)
    (out_dir / "migration_report.md").write_text(report_md, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"MIGRATION DRY-RUN STATUS: {result.status}")
    print(f"Artifacts saved to: {out_dir.resolve()}")
    print("=" * 60)

    if result.status == "VERIFIED":
        return 0
    elif result.status == "UNVERIFIED":
        return 2
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
