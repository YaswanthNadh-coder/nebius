"""Unit tests for key loading, clean_sql, model resolution, sandbox, and oracle transforms."""

import os
import sqlite3
from unittest.mock import MagicMock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from timemachine.adapters.schema import ColumnInfo, TableSchema
from timemachine.agent.llm import clean_sql, resolve_nebius_models
from timemachine.config import (
    is_placeholder_key,
    load_env_file,
)
from timemachine.traps.generator import generate_trap_rows
from timemachine.verify.oracle import (
    cast_cents,
    cast_int,
    default_if_null,
    map_via_lookup,
    normalize_date,
    split_name_first,
    split_name_last,
    strip_honorific,
)
from timemachine.verify.sandbox import (
    apply_sandbox,
)

# =========================================================================
# 1. Clean SQL Tests
# =========================================================================

def test_clean_sql_tagged_fences():
    sql1 = "```sql\nSELECT 1;\n```"
    assert clean_sql(sql1) == "SELECT 1;"

    sql2 = "```sqlite\nSELECT 2;\n```"
    assert clean_sql(sql2) == "SELECT 2;"

    sql3 = "```\nSELECT 3;\n```"
    assert clean_sql(sql3) == "SELECT 3;"


def test_clean_sql_think_tags():
    raw_closed = "<think>Planning migration steps...</think>\n```sql\nALTER TABLE t ADD COLUMN c TEXT;\n```"
    assert clean_sql(raw_closed) == "ALTER TABLE t ADD COLUMN c TEXT;"

    raw_unclosed = "<think>Unfinished thoughts\nSELECT * FROM t;"
    assert clean_sql(raw_unclosed) == ""


def test_clean_sql_multiple_blocks():
    raw = (
        "Here is the forward migration:\n"
        "```sql\nCREATE TABLE t2 (id INT);\n```\n"
        "And the insert:\n"
        "```sql\nINSERT INTO t2 SELECT id FROM t;\n```"
    )
    res = clean_sql(raw)
    assert "CREATE TABLE t2" in res
    assert "INSERT INTO t2" in res


def test_clean_sql_unterminated_fence_and_prose():
    raw = (
        "Sure, here is your migration script:\n"
        "```sql\n"
        "CREATE TABLE new_t (id INT PRIMARY KEY);\n"
        "INSERT INTO new_t SELECT id FROM old_t;"
    )
    res = clean_sql(raw)
    assert res.startswith("CREATE TABLE new_t")
    assert "INSERT INTO new_t" in res


def test_clean_sql_empty_and_none():
    assert clean_sql(None) == ""
    assert clean_sql("") == ""
    assert clean_sql("    ") == ""
    assert clean_sql("Just casual conversation without any SQL keywords.") == ""


# =========================================================================
# 2. Key Loading Tests
# =========================================================================

def test_is_placeholder_key():
    assert is_placeholder_key("") is True
    assert is_placeholder_key("   ") is True
    assert is_placeholder_key(None) is True
    assert is_placeholder_key("your_nebius_api_key_here") is True
    assert is_placeholder_key("your_key_123") is True
    assert is_placeholder_key('"your_key"') is True
    assert is_placeholder_key("'your_key'") is True
    assert is_placeholder_key("live_api_key_abc12345") is False


def test_load_env_bom_and_stale_override(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    monkeypatch.setenv("NEBIUS_API_KEY", "stale_placeholder_key")

    # Write UTF-8 with BOM
    content = '\ufeffNEBIUS_API_KEY="new_active_key_999"\n'
    env_file.write_bytes(content.encode("utf-8"))

    loaded_key = load_env_file(str(env_file))
    assert loaded_key == "new_active_key_999"
    assert os.environ.get("NEBIUS_API_KEY") == "new_active_key_999"


def test_load_env_utf16_tolerance(tmp_path):
    env_file = tmp_path / ".env"
    content = 'NEBIUS_API_KEY="utf16_active_key_888"\n'
    env_file.write_bytes(content.encode("utf-16"))

    loaded_key = load_env_file(str(env_file))
    assert loaded_key == "utf16_active_key_888"


# =========================================================================
# 3. Model Resolution Tests
# =========================================================================

def test_resolve_nebius_models_prefers_exact_defaults():
    client = MagicMock()
    m1 = MagicMock(id="nvidia/nemotron-3-super-120b-a12b")
    m2 = MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b")
    m3 = MagicMock(id="nebius/bge-multilingual-reranker")
    m4 = MagicMock(id="meta/llama-guard-3")
    client.models.list.return_value.data = [m3, m4, m1, m2]

    fast, ultra, warns = resolve_nebius_models(client)
    assert fast == "nvidia/nemotron-3-super-120b-a12b"
    assert ultra == "nvidia/Nemotron-3-Ultra-550b-a55b"
    assert len(warns) == 0


def test_resolve_nebius_models_env_overrides(monkeypatch):
    client = MagicMock()
    monkeypatch.setenv("NEBIUS_FAST_MODEL", "custom-fast-model")
    monkeypatch.setenv("NEBIUS_ULTRA_MODEL", "custom-ultra-model")

    fast, ultra, _ = resolve_nebius_models(client)
    assert fast == "custom-fast-model"
    assert ultra == "custom-ultra-model"


def test_resolve_nebius_models_nano_fallback():
    client = MagicMock()
    m1 = MagicMock(id="nvidia/nemotron-nano-8b")
    m2 = MagicMock(id="nvidia/Nemotron-3-Ultra-550b-a55b")
    client.models.list.return_value.data = [m1, m2]

    fast, ultra, warns = resolve_nebius_models(client)
    assert fast == "nvidia/nemotron-nano-8b"
    assert ultra == "nvidia/Nemotron-3-Ultra-550b-a55b"
    assert any("falling back to Nano" in w for w in warns)


# =========================================================================
# 4. Sandbox Authorizer and Timeout Tests
# =========================================================================

def test_sandbox_authorizer_denies_attach_and_vacuum(tmp_path):
    conn = sqlite3.connect(":memory:")
    apply_sandbox(conn, timeout_sec=2.0)

    # Normal DDL and DML should succeed
    conn.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT);")
    conn.execute("INSERT INTO t VALUES (1, 'Alice');")
    conn.commit()

    # ATTACH must be denied
    with pytest.raises(sqlite3.DatabaseError) as exc:
        conn.execute(f"ATTACH DATABASE '{tmp_path}/rogue.db' AS rogue;")
    assert "authoriz" in str(exc.value).lower() or "not authorized" in str(exc.value).lower()

    # VACUUM INTO must be denied
    with pytest.raises(sqlite3.DatabaseError) as exc_v:
        conn.execute(f"VACUUM INTO '{tmp_path}/dump.db';")
    assert "authoriz" in str(exc_v.value).lower() or "not authorized" in str(exc_v.value).lower()

    # Deny unauthorized PRAGMAs
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("PRAGMA writable_schema = ON;")

    # Allowed PRAGMAs should succeed
    cur = conn.execute("PRAGMA table_info(t);")
    assert len(cur.fetchall()) == 2
    cur2 = conn.execute("PRAGMA foreign_key_check;")
    assert len(cur2.fetchall()) == 0

    conn.close()


def test_sandbox_progress_handler_blocks_infinite_recursion():
    conn = sqlite3.connect(":memory:")
    # Set tight 0.5s timeout for fast test
    apply_sandbox(conn, timeout_sec=0.5)

    conn.execute("CREATE TABLE t (n INT);")
    conn.execute("INSERT INTO t VALUES (1);")

    # Infinite CTE recursion
    infinite_sql = """
    WITH RECURSIVE inf(x) AS (
        SELECT 1
        UNION ALL
        SELECT x + 1 FROM inf
    )
    SELECT count(*) FROM inf;
    """
    with pytest.raises(sqlite3.OperationalError) as exc:
        conn.execute(infinite_sql)
    assert "interrupted" in str(exc.value).lower() or "operation" in str(exc.value).lower()
    conn.close()


# =========================================================================
# 5. Oracle Transform Tests
# =========================================================================

def test_oracle_name_splitting():
    assert split_name_first("Jane Doe") == "Jane"
    assert split_name_last("Jane Doe") == "Doe"

    # Mononym / single name
    assert split_name_first("Madonna") == "Madonna"
    assert split_name_last("Madonna") == "N/A"

    # Multiple spaces / middle names
    assert split_name_first("John Fitzgerald Kennedy") == "John"
    assert split_name_last("John Fitzgerald Kennedy") == "Fitzgerald Kennedy"

    # Whitespace padding
    assert split_name_first("   Bob   Marley   ") == "Bob"
    assert split_name_last("   Bob   Marley   ") == "Marley"


def test_oracle_strip_honorific():
    assert strip_honorific("Dr. Jane Doe") == "Jane Doe"
    assert strip_honorific("Mr. Bob Smith") == "Bob Smith"
    assert strip_honorific("Prof. Charles Xavier") == "Charles Xavier"
    assert strip_honorific("Alice Wonder") == "Alice Wonder"


def test_oracle_cast_int_and_cents():
    assert cast_int("$120,000") == 120000
    assert cast_int("  $45,500.80  ") == 45500
    assert cast_int("- $500") == -500
    assert cast_int(None, default=0) == 0

    assert cast_cents("$120.50") == 12050
    assert cast_cents("85") == 8500
    assert cast_cents("$0.99") == 99


def test_oracle_default_if_null():
    assert default_if_null(None, "UNKNOWN") == "UNKNOWN"
    assert default_if_null("", "UNKNOWN") == "UNKNOWN"
    assert default_if_null("   ", "UNKNOWN") == "UNKNOWN"
    assert default_if_null("ENG", "UNKNOWN") == "ENG"


def test_oracle_map_via_lookup():
    lookup = {"ENG": "Engineering", "HR": "Human Resources"}
    assert map_via_lookup("ENG", lookup, "Other") == "Engineering"
    assert map_via_lookup("LEGACY_99", lookup, "UNASSIGNED") == "UNASSIGNED"
    assert map_via_lookup(None, lookup, "UNASSIGNED") == "UNASSIGNED"


def test_oracle_normalize_date():
    assert normalize_date("2026-03-15") == "2026-03-15"
    assert normalize_date("15/03/2026") == "2026-03-15"
    assert normalize_date("03/15/2026") == "2026-03-15"
    assert normalize_date("March 15, 2026") == "2026-03-15"


# =========================================================================
# 6. Trap Generator Property Tests (Hypothesis)
# =========================================================================

@given(st.text(min_size=1, max_size=20))
def test_trap_generator_does_not_crash(col_name):
    tbl = TableSchema(
        name="test_table",
        columns=[
            ColumnInfo(name="id", type="INTEGER", pk=True),
            ColumnInfo(name=col_name.replace('"', ""), type="TEXT", notnull=False),
        ],
    )
    traps = generate_trap_rows(tbl, start_pk=1000)
    assert len(traps) > 0
    for r in traps:
        assert isinstance(r["pk"], int)
        assert isinstance(r["trap_type"], str)
        assert isinstance(r["row_dict"], dict)
