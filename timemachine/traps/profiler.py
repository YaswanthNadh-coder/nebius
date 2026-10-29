"""Schema-driven database profiler with privacy-preserving masking."""

import re
import sqlite3
from typing import Any

from pydantic import BaseModel, Field

from timemachine.adapters.schema import DatabaseSchema, TableSchema


class ColumnProfile(BaseModel):
    """Statistical and pattern profile of a single column."""

    model_config = {"populate_by_name": True}

    name: str = Field(default="", alias="column_name")
    column_type: str = "TEXT"
    total_rows: int = Field(default=0, alias="total_count")
    null_count: int = 0
    null_rate: float = 0.0
    distinct_count: int = 0
    min_length: int | None = None
    max_length: int | None = None
    has_unicode: bool = False
    has_currency: bool = False
    has_whitespace_padding: bool = False
    has_case_variants: bool = False
    orphan_fk_count: int = 0
    is_nullable: bool = True
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


MAX_SAMPLE_ROWS = 50000

_PROFILES_CACHE: dict[tuple[str, bool], dict[str, dict[str, ColumnProfile]]] = {}


def get_cached_profile(fingerprint: str, inject_traps: bool = True) -> dict[str, dict[str, ColumnProfile]] | None:
    """Retrieve cached profile for (fingerprint, inject_traps) tuple."""
    return _PROFILES_CACHE.get((fingerprint, inject_traps))


def set_cached_profile(
    fingerprint: str,
    inject_traps: bool,
    profile: dict[str, dict[str, ColumnProfile]],
) -> None:
    """Store profile in memory cache keyed by (fingerprint, inject_traps)."""
    _PROFILES_CACHE[(fingerprint, inject_traps)] = profile


def clear_profiles_cache() -> None:
    """Clear memory profiles cache."""
    _PROFILES_CACHE.clear()


def profile_table(conn: sqlite3.Connection, table_schema: TableSchema | str, limit_samples: int = 5) -> dict[str, ColumnProfile]:
    """Inspect columns in table, compute aggregate statistics and generate masked samples in one consolidated query."""
    cur = conn.cursor()
    if isinstance(table_schema, str):
        from timemachine.adapters.schema import ColumnInfo
        cur.execute(f'PRAGMA table_info("{table_schema}")')
        cols = [ColumnInfo(name=r[1], type=r[2], notnull=bool(r[3]), pk=bool(r[5])) for r in cur.fetchall()]
        table_schema = TableSchema(name=table_schema, columns=cols)
    tname = table_schema.name
    profiles: dict[str, ColumnProfile] = {}

    cur.execute(f"SELECT COUNT(*) FROM \"{tname}\"")
    total_rows = cur.fetchone()[0]

    use_sampling = total_rows > MAX_SAMPLE_ROWS
    table_src = f'(SELECT * FROM "{tname}" LIMIT {MAX_SAMPLE_ROWS})' if use_sampling else f'"{tname}"'
    sample_size = min(total_rows, MAX_SAMPLE_ROWS) if use_sampling else total_rows

    if not table_schema.columns:
        return profiles

    # Compute stats for all columns in one consolidated query
    agg_clauses = []
    for col in table_schema.columns:
        cname = col.name
        agg_clauses.extend([
            f'COUNT("{cname}")',
            f'COUNT(DISTINCT "{cname}")',
            f'MIN(LENGTH(CAST("{cname}" AS TEXT)))',
            f'MAX(LENGTH(CAST("{cname}" AS TEXT)))',
        ])
    cur.execute(f"SELECT {', '.join(agg_clauses)} FROM {table_src}")
    agg_row = cur.fetchone()

    # Fetch a sample batch of rows across all columns for pattern analysis
    col_names = [c.name for c in table_schema.columns]
    quoted_cols = ", ".join(f'"{c}"' for c in col_names)
    cur.execute(f"SELECT {quoted_cols} FROM {table_src} LIMIT 50")
    sample_matrix = cur.fetchall()

    for idx, col in enumerate(table_schema.columns):
        cname = col.name
        base_idx = idx * 4
        non_null_cnt = agg_row[base_idx] if agg_row else 0
        distinct_cnt = agg_row[base_idx + 1] if agg_row else 0
        min_len = agg_row[base_idx + 2] if agg_row else None
        max_len = agg_row[base_idx + 3] if agg_row else None

        sample_null_cnt = sample_size - non_null_cnt
        null_rate = round(sample_null_cnt / sample_size, 4) if sample_size > 0 else 0.0
        null_cnt = int(round(null_rate * total_rows)) if use_sampling else sample_null_cnt

        sample_vals = [str(row[idx]) for row in sample_matrix if row[idx] is not None]

        has_unicode = any(ord(char) > 127 for val in sample_vals for char in val)
        has_currency = any(any(c in val for c in ("$", "€", "£", "¥")) for val in sample_vals)
        has_whitespace = any(val != val.strip() for val in sample_vals)

        # Check orphan FK count if col is in table_schema.foreign_keys
        orphan_cnt = 0
        for fk in table_schema.foreign_keys:
            if fk.from_col.lower() == cname.lower():
                try:
                    cur.execute(
                        f"SELECT COUNT(*) FROM {table_src} t "
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
            has_case_variants=False,
            orphan_fk_count=orphan_cnt,
            is_nullable=not col.notnull,
            masked_sample=masked,
        )

    return profiles


def profile_database(conn: sqlite3.Connection, schema: DatabaseSchema) -> dict[str, dict[str, ColumnProfile]]:
    """Profile all tables in database."""
    full_profile: dict[str, dict[str, ColumnProfile]] = {}
    for tname, tbl in schema.tables.items():
        full_profile[tname] = profile_table(conn, tbl)
    return full_profile
