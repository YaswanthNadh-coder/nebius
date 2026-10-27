"""Schema-driven database profiler with privacy-preserving masking."""

import re
import sqlite3
from typing import Any

from pydantic import BaseModel, Field

from timemachine.adapters.schema import DatabaseSchema, TableSchema


class ColumnProfile(BaseModel):
    """Statistical and pattern profile of a single column."""

    name: str
    column_type: str
    total_rows: int
    null_count: int
    null_rate: float
    distinct_count: int
    min_length: int | None = None
    max_length: int | None = None
    has_unicode: bool = False
    has_currency: bool = False
    has_whitespace_padding: bool = False
    has_case_variants: bool = False
    orphan_fk_count: int = 0
    masked_sample: list[str] = Field(default_factory=list)


def mask_value(val: Any) -> str:
    """Mask PII (emails, phone numbers, names, IDs) for privacy."""
    if val is None:
        return "NULL"
    s = str(val).strip()
    if not s:
        return '""'

    # Email pattern: test@example.com -> t***@e***.com
    if "@" in s and "." in s:
        parts = s.split("@", 1)
        user = parts[0][0] + "***" if len(parts[0]) > 0 else "***"
        domain_parts = parts[1].split(".", 1)
        dom = domain_parts[0][0] + "***" if len(domain_parts[0]) > 0 else "***"
        tld = domain_parts[1] if len(domain_parts) > 1 else "com"
        return f"{user}@{dom}.{tld}"

    # Phone / number sequence: +1-555-1234 -> +1-***-***
    if re.search(r"\d{3,}", s) and re.search(r"[-+.()\s\d]{7,}", s):
        return re.sub(r"\d", "*", s)

    # General string: retain length shape and first/last char
    if len(s) <= 2:
        return "**"
    return f"{s[0]}{'*' * min(len(s) - 2, 4)}{s[-1]}"


def profile_table(conn: sqlite3.Connection, table_schema: TableSchema, limit_samples: int = 5) -> dict[str, ColumnProfile]:
    """Inspect columns in table, compute aggregate statistics and generate masked samples."""
    cur = conn.cursor()
    tname = table_schema.name
    profiles: dict[str, ColumnProfile] = {}

    cur.execute(f"SELECT COUNT(*) FROM \"{tname}\"")
    total_rows = cur.fetchone()[0]

    for col in table_schema.columns:
        cname = col.name
        cur.execute(f"SELECT COUNT(*) FROM \"{tname}\" WHERE \"{cname}\" IS NULL")
        null_cnt = cur.fetchone()[0]
        null_rate = round(null_cnt / total_rows, 4) if total_rows > 0 else 0.0

        cur.execute(f"SELECT COUNT(DISTINCT \"{cname}\") FROM \"{tname}\"")
        distinct_cnt = cur.fetchone()[0]

        # String length and pattern inspection
        cur.execute(
            f"SELECT MIN(LENGTH(CAST(\"{cname}\" AS TEXT))), MAX(LENGTH(CAST(\"{cname}\" AS TEXT))) "
            f"FROM \"{tname}\" WHERE \"{cname}\" IS NOT NULL"
        )
        min_len, max_len = cur.fetchone()

        # Check for currency symbols, unicode, whitespace padding
        cur.execute(
            f"SELECT \"{cname}\" FROM \"{tname}\" WHERE \"{cname}\" IS NOT NULL LIMIT 50"
        )
        sample_vals = [str(r[0]) for r in cur.fetchall()]

        has_unicode = any(ord(char) > 127 for val in sample_vals for char in val)
        has_currency = any(any(c in val for c in ("$", "€", "£", "¥")) for val in sample_vals)
        has_whitespace = any(val != val.strip() for val in sample_vals)

        # Check case-variant duplicates
        cur.execute(
            f"SELECT COUNT(*) FROM ("
            f"  SELECT LOWER(CAST(\"{cname}\" AS TEXT)) FROM \"{tname}\" "
            f"  WHERE \"{cname}\" IS NOT NULL GROUP BY LOWER(CAST(\"{cname}\" AS TEXT)) "
            f"  HAVING COUNT(DISTINCT CAST(\"{cname}\" AS TEXT)) > 1"
            f")"
        )
        case_var_cnt = cur.fetchone()[0]
        has_case_variants = case_var_cnt > 0

        # Check orphan FK count if col is in table_schema.foreign_keys
        orphan_cnt = 0
        for fk in table_schema.foreign_keys:
            if fk.from_col.lower() == cname.lower():
                try:
                    cur.execute(
                        f"SELECT COUNT(*) FROM \"{tname}\" t "
                        f"WHERE t.\"{cname}\" IS NOT NULL "
                        f"AND NOT EXISTS (SELECT 1 FROM \"{fk.to_table}\" r WHERE r.\"{fk.to_col}\" = t.\"{cname}\")"
                    )
                    orphan_cnt = cur.fetchone()[0]
                except Exception:
                    pass

        masked = [mask_value(v) for v in sample_vals[:limit_samples]]

        profiles[cname] = ColumnProfile(
            name=cname,
            column_type=col.type,
            total_rows=total_rows,
            null_count=null_cnt,
            null_rate=null_rate,
            distinct_count=distinct_cnt,
            min_length=min_len,
            max_length=max_len,
            has_unicode=has_unicode,
            has_currency=has_currency,
            has_whitespace_padding=has_whitespace,
            has_case_variants=has_case_variants,
            orphan_fk_count=orphan_cnt,
            masked_sample=masked,
        )

    return profiles


def profile_database(conn: sqlite3.Connection, schema: DatabaseSchema) -> dict[str, dict[str, ColumnProfile]]:
    """Profile all tables in database."""
    full_profile: dict[str, dict[str, ColumnProfile]] = {}
    for tname, tbl in schema.tables.items():
        full_profile[tname] = profile_table(conn, tbl)
    return full_profile
