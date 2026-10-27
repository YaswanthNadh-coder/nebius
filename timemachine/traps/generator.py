"""Adversarial production edge-case generator and trap tracker."""

import sqlite3
from typing import Any

from pydantic import BaseModel

from timemachine.adapters.schema import DatabaseSchema, TableSchema
from timemachine.logging import get_logger

logger = get_logger("traps.generator")

TRAP_TABLE_NAME = "__trap_rows"


class TrapRecord(BaseModel):
    """Metadata for an injected adversarial test row."""

    table_name: str
    pk_value: str
    trap_type: str
    description: str


TRAP_DEFINITIONS = [
    ("NULL_IN_NULLABLE", "NULL value in nullable column"),
    ("EMPTY_STRING", "Empty string in text column"),
    ("WHITESPACE_PADDING", "Leading/trailing spaces and tabs around text"),
    ("UNICODE_EMOJI", "Apostrophes, emojis and non-ASCII unicode (O'Reilly 🚀)"),
    ("SINGLE_WORD_NAME", "Mononym name with no space separator (Madonna)"),
    ("SQL_PUNCTUATION", "Quotes, semicolons and SQL-like text"),
    ("CURRENCY_STRING", "Formatted currency with dollar sign and commas ($125,000)"),
    ("MIXED_DATE_FORMAT", "Mixed non-ISO date string (10/25/2026)"),
    ("ORPHAN_FOREIGN_KEY", "Unmapped legacy foreign key reference (LEGACY_99)"),
    ("CASE_VARIANT", "Case-sensitive duplicate variant ('Active' vs 'active')"),
    ("VERY_LONG_STRING", "String with 500 characters testing buffer boundaries"),
]


def ensure_trap_table(conn: sqlite3.Connection) -> None:
    """Ensure the __trap_rows metadata table exists."""
    cur = conn.cursor()
    cur.execute(
        f"CREATE TABLE IF NOT EXISTS {TRAP_TABLE_NAME} ("
        "  table_name TEXT NOT NULL,"
        "  pk_value TEXT NOT NULL,"
        "  trap_type TEXT NOT NULL,"
        "  description TEXT"
        ")"
    )
    conn.commit()


def get_next_pk(conn: sqlite3.Connection, table_name: str, pk_col: str) -> int:
    """Get the next integer ID for a table."""
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT MAX(\"{pk_col}\") FROM \"{table_name}\"")
        row = cur.fetchone()
        return (row[0] or 0) + 1
    except Exception:
        return 999000


def inject_traps_for_table(
    conn: sqlite3.Connection,
    table: TableSchema,
    start_pk: int | None = None,
) -> list[TrapRecord]:
    """Inject edge-case rows for a specific table into the database."""
    ensure_trap_table(conn)
    cur = conn.cursor()
    tname = table.name
    pks = table.pk_columns

    if not pks:
        return []

    pk_col = pks[0]
    next_pk = start_pk if start_pk is not None else get_next_pk(conn, tname, pk_col)
    records: list[TrapRecord] = []

    # Generate test cases
    test_cases = [
        # (trap_type, desc, value_modifier_dict)
        (
            "NULL_IN_NULLABLE",
            "NULL values in non-PK nullable columns",
            {c.name: None for c in table.columns if not c.notnull and c.name.lower() != pk_col.lower()},
        ),
        (
            "SINGLE_WORD_NAME",
            "Mononym name without space delimiter (Madonna)",
            {
                c.name: "Madonna"
                for c in table.columns
                if "name" in c.name.lower() and c.name.lower() != pk_col.lower()
            },
        ),
        (
            "UNICODE_EMOJI",
            "Apostrophes and emojis (O'Reilly 🚀)",
            {
                c.name: "O'Reilly 🚀"
                for c in table.columns
                if "name" in c.name.lower() or "text" in c.name.lower()
            },
        ),
        (
            "WHITESPACE_PADDING",
            "Leading and trailing whitespace padding",
            {
                c.name: "   Dr. Grace Hopper   \t"
                for c in table.columns
                if "name" in c.name.lower() or "text" in c.name.lower()
            },
        ),
        (
            "CURRENCY_STRING",
            "Dollar prefixed salary string with commas ($145,000)",
            {
                c.name: "$145,000"
                for c in table.columns
                if "salary" in c.name.lower() or "price" in c.name.lower() or "amount" in c.name.lower()
            },
        ),
        (
            "ORPHAN_FOREIGN_KEY",
            "Unmapped legacy foreign key code (LEGACY_99)",
            {
                fk.from_col: "LEGACY_99"
                for fk in table.foreign_keys
            },
        ),
        (
            "SQL_PUNCTUATION",
            "Quotes, semicolon and injection text",
            {
                c.name: "Robert'); DROP TABLE Students;--"
                for c in table.columns
                if ("name" in c.name.lower() or "text" in c.name.lower()) and c.name.lower() != pk_col.lower()
            },
        ),
        (
            "VERY_LONG_STRING",
            "500-character long boundary test string",
            {
                c.name: "A" * 500
                for c in table.columns
                if ("name" in c.name.lower() or "text" in c.name.lower()) and c.name.lower() != pk_col.lower()
            },
        ),
    ]

    for trap_type, desc, mod_values in test_cases:
        if not mod_values:
            continue

        row_data: dict[str, Any] = {}
        for col in table.columns:
            cname = col.name
            if cname.lower() == pk_col.lower():
                row_data[cname] = next_pk
            elif cname in mod_values:
                row_data[cname] = mod_values[cname]
            else:
                # Default benign value based on type
                ctype = col.type.upper()
                if "INT" in ctype:
                    row_data[cname] = 1
                elif "REAL" in ctype or "FLOA" in ctype or "DOUB" in ctype:
                    row_data[cname] = 1.0
                else:
                    row_data[cname] = f"TestVal_{next_pk}"

        col_names = list(row_data.keys())
        placeholders = ", ".join(["?"] * len(col_names))
        quoted_cols = ", ".join([f"\"{c}\"" for c in col_names])
        insert_sql = f"INSERT OR IGNORE INTO \"{tname}\" ({quoted_cols}) VALUES ({placeholders})"

        try:
            cur.execute(insert_sql, [row_data[c] for c in col_names])
            cur.execute(
                f"INSERT INTO {TRAP_TABLE_NAME} (table_name, pk_value, trap_type, description) VALUES (?, ?, ?, ?)",
                (tname, str(next_pk), trap_type, desc),
            )
            records.append(
                TrapRecord(
                    table_name=tname,
                    pk_value=str(next_pk),
                    trap_type=trap_type,
                    description=desc,
                )
            )
            next_pk += 1
        except Exception as exc:
            logger.debug("Trap injection notice for %s: %s", trap_type, exc)

    conn.commit()
    return records


def inject_adversarial_traps(conn: sqlite3.Connection, schema: DatabaseSchema) -> list[TrapRecord]:
    """Inject edge-case traps into all tables in schema."""
    all_records: list[TrapRecord] = []
    for tbl in schema.tables.values():
        all_records.extend(inject_traps_for_table(conn, tbl))
    return all_records


def audit_traps(conn: sqlite3.Connection, table_name: str, pk_col: str = "id") -> list[dict[str, Any]]:
    """Audit which trap rows survived and evaluate their integrity."""
    cur = conn.cursor()
    # Check if trap table exists
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (TRAP_TABLE_NAME,),
    )
    if not cur.fetchone():
        return []

    cur.execute(
        f"SELECT pk_value, trap_type, description FROM {TRAP_TABLE_NAME} WHERE table_name=?",
        (table_name,),
    )
    traps = cur.fetchall()

    results: list[dict[str, Any]] = []
    for pk_val, trap_type, desc in traps:
        try:
            cur.execute(f"SELECT * FROM \"{table_name}\" WHERE \"{pk_col}\" = ?", (pk_val,))
            row = cur.fetchone()
            survived = row is not None
            results.append({
                "pk": pk_val,
                "trap_type": trap_type,
                "description": desc,
                "survived": survived,
                "status": "PASSED" if survived else "LOST",
            })
        except Exception as exc:
            results.append({
                "pk": pk_val,
                "trap_type": trap_type,
                "description": desc,
                "survived": False,
                "status": f"ERROR: {exc}",
            })

    return results


def generate_trap_rows(table: TableSchema, start_pk: int = 1000) -> list[dict[str, Any]]:
    """Pure in-memory generation of trap records for schema-driven property testing."""
    pks = table.pk_columns
    pk_col = pks[0] if pks else "id"
    records: list[dict[str, Any]] = []

    test_cases = [
        ("NULL_IN_NULLABLE", "NULL values in non-PK nullable columns", {c.name: None for c in table.columns if not c.notnull and c.name.lower() != pk_col.lower()}),
        ("SINGLE_WORD_NAME", "Mononym name without space delimiter", {c.name: "Madonna" for c in table.columns if "name" in c.name.lower() and c.name.lower() != pk_col.lower()}),
        ("UNICODE_EMOJI", "Apostrophes and emojis", {c.name: "O'Reilly 🚀" for c in table.columns if "name" in c.name.lower() or "text" in c.name.lower()}),
        ("WHITESPACE_PADDING", "Leading and trailing whitespace", {c.name: "   Dr. Grace Hopper   \t" for c in table.columns if "name" in c.name.lower() or "text" in c.name.lower()}),
        ("CURRENCY_STRING", "Formatted currency string", {c.name: "$145,000" for c in table.columns if "salary" in c.name.lower() or "price" in c.name.lower()}),
        ("VERY_LONG_STRING", "500-character test string", {c.name: "A" * 500 for c in table.columns if ("name" in c.name.lower() or "text" in c.name.lower()) and c.name.lower() != pk_col.lower()}),
    ]

    cur_pk = start_pk
    for trap_type, desc, mod_vals in test_cases:
        row_dict: dict[str, Any] = {}
        for col in table.columns:
            cname = col.name
            if cname.lower() == pk_col.lower():
                row_dict[cname] = cur_pk
            elif cname in mod_vals:
                row_dict[cname] = mod_vals[cname]
            else:
                row_dict[cname] = "test"
        records.append({"pk": cur_pk, "trap_type": trap_type, "description": desc, "row_dict": row_dict})
        cur_pk += 1

    return records


def inject_traps_into_clone(
    conn: sqlite3.Connection,
    table: TableSchema | str,
    schema: DatabaseSchema | None = None,
) -> list[TrapRecord]:
    """Inject edge-case traps into a database clone for a table or schema."""
    if isinstance(table, str):
        if schema and table in schema.tables:
            return inject_traps_for_table(conn, schema.tables[table])
        return []
    return inject_traps_for_table(conn, table)
