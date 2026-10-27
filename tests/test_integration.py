"""Integration tests on a real SQLite fixture with foreign keys, indexes, views, and triggers."""

import hashlib
import sqlite3

import pytest

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.verify.invariants import evaluate_migration_invariants
from timemachine.verify.rebuild import verify_rollback_roundtrip
from timemachine.verify.spec import ColumnMapping, MigrationSpec


@pytest.fixture
def chinook_fixture(tmp_path):
    """Create a self-contained Chinook-like database fixture with indexes, FKs, views, and triggers."""
    db_path = tmp_path / "chinook_mini.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Tables with Foreign Keys
    cur.execute("""
        CREATE TABLE artists (
            ArtistId INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE albums (
            AlbumId INTEGER PRIMARY KEY AUTOINCREMENT,
            Title TEXT NOT NULL,
            ArtistId INTEGER,
            FOREIGN KEY (ArtistId) REFERENCES artists(ArtistId)
        );
    """)
    cur.execute("""
        CREATE TABLE customers (
            CustomerId INTEGER PRIMARY KEY AUTOINCREMENT,
            FirstName TEXT NOT NULL,
            LastName TEXT NOT NULL,
            Company TEXT,
            Country TEXT,
            Phone TEXT
        );
    """)

    # Indexes
    cur.execute("CREATE INDEX IFK_AlbumArtistId ON albums(ArtistId);")
    cur.execute("CREATE UNIQUE INDEX idx_customer_phone ON customers(Phone);")

    # View
    cur.execute("""
        CREATE VIEW v_album_artists AS
        SELECT a.AlbumId, a.Title, art.Name AS Artist
        FROM albums a
        JOIN artists art ON a.ArtistId = art.ArtistId;
    """)

    # Trigger
    cur.execute("""
        CREATE TRIGGER trg_album_audit AFTER INSERT ON albums
        BEGIN
            SELECT 1;
        END;
    """)

    # Seed data
    cur.executemany(
        "INSERT INTO artists (ArtistId, Name) VALUES (?, ?);",
        [(1, "AC/DC"), (2, "Accept"), (3, "Aerosmith")],
    )
    cur.executemany(
        "INSERT INTO albums (AlbumId, Title, ArtistId) VALUES (?, ?, ?);",
        [
            (1, "For Those About To Rock", 1),
            (2, "Balls to the Wall", 2),
            (3, "Restless and Wild", 2),
            (4, "Orphan Album Without Artist", 9999),  # Pre-existing FK violation!
        ],
    )
    cur.executemany(
        "INSERT INTO customers (CustomerId, FirstName, LastName, Company, Country, Phone) VALUES (?, ?, ?, ?, ?, ?);",
        [
            (1, "Luís", "Gonçalves", "Embraer - Empresa Brasileira", "Brazil", "+55 (12) 3923-5555"),
            (2, "Leonie", "Köhler", None, "Germany", "+49 0711 2842222"),
            (3, "François", "Tremblay", None, "Canada", "+1 (514) 721-4711"),
            (4, "Bjørn", "Hansen", None, "Norway", "+47 22 44 22 22"),
        ],
    )

    conn.commit()
    conn.close()
    return str(db_path)


def test_source_db_file_byte_identical_preservation(chinook_fixture):
    """Verify that source database file is 100% byte-identical before and after adapter operations."""
    with open(chinook_fixture, "rb") as f:
        sha_before = hashlib.sha256(f.read()).hexdigest()

    adapter = SQLiteAdapter.from_file(chinook_fixture)
    schema = adapter.introspect()
    assert "customers" in schema.tables
    assert "albums" in schema.tables

    # Run a migration on a disposable clone
    clone = adapter.clone()
    ok, err, _ = adapter.run_in_sandbox(
        clone,
        "ALTER TABLE customers ADD COLUMN status TEXT DEFAULT 'active';",
        fk_enabled=True,
    )
    assert ok is True
    clone.close()

    # Re-verify source file SHA256
    with open(chinook_fixture, "rb") as f:
        sha_after = hashlib.sha256(f.read()).hexdigest()

    assert sha_before == sha_after, "Source database file was modified during dry-run!"


def test_table_rebuild_preserves_indexes_triggers_views(chinook_fixture):
    """SQLite 12-step table rebuild must preserve all indexes, triggers, and views."""
    adapter = SQLiteAdapter.from_file(chinook_fixture)
    base_conn = adapter.open_readonly()
    migrated_conn = adapter.clone()

    # SQLite rebuild procedure on customers: make Company NOT NULL with default 'N/A'
    rebuild_sql = """
    PRAGMA foreign_keys = OFF;
    CREATE TABLE customers_new (
        CustomerId INTEGER PRIMARY KEY AUTOINCREMENT,
        FirstName TEXT NOT NULL,
        LastName TEXT NOT NULL,
        Company TEXT NOT NULL DEFAULT 'N/A',
        Country TEXT,
        Phone TEXT
    );
    INSERT INTO customers_new (CustomerId, FirstName, LastName, Company, Country, Phone)
    SELECT CustomerId, FirstName, LastName, COALESCE(Company, 'N/A'), Country, Phone
    FROM customers;
    DROP TABLE customers;
    ALTER TABLE customers_new RENAME TO customers;
    CREATE UNIQUE INDEX idx_customer_phone ON customers(Phone);
    PRAGMA foreign_keys = ON;
    """

    ok, err, elapsed = adapter.run_in_sandbox(migrated_conn, rebuild_sql, fk_enabled=True)
    assert ok is True, f"Migration failed: {err}"

    spec = MigrationSpec(
        affected_tables=["customers"],
        column_mappings={
            "customers": [
                ColumnMapping(new_column="Company", source_column="Company", transform="default_if_null", default_value="N/A"),
            ]
        },
        row_policy="preserve_all",
        columns_declared_unchanged={"customers": ["CustomerId", "FirstName", "LastName", "Country", "Phone"]},
    )

    inv = evaluate_migration_invariants(
        base_conn,
        migrated_conn,
        target_table="customers",
        spec=spec,
        elapsed_ms=elapsed,
    )

    assert inv.passed is True
    assert inv.status == "VERIFIED"
    assert inv.before_count == 4
    assert inv.after_count == 4
    assert any("idx_customer_phone" in c for c in inv.passed_checks)


def test_foreign_keys_tested_both_on_and_off(chinook_fixture):
    """Migration must be audited under both foreign_keys=ON and foreign_keys=OFF."""
    adapter = SQLiteAdapter.from_file(chinook_fixture)

    migration_sql = "ALTER TABLE customers ADD COLUMN notes TEXT DEFAULT '';"

    # Test under FK ON
    c_on = adapter.clone()
    ok_on, _, _ = adapter.run_in_sandbox(c_on, migration_sql, fk_enabled=True)
    assert ok_on is True
    c_on.close()

    # Test under FK OFF
    c_off = adapter.clone()
    ok_off, _, _ = adapter.run_in_sandbox(c_off, migration_sql, fk_enabled=False)
    assert ok_off is True
    c_off.close()


def test_preexisting_fk_violations_baseline_subtracted(chinook_fixture):
    """Pre-existing orphan rows (Album 4 -> Artist 9999) must NOT be blamed on the migration."""
    adapter = SQLiteAdapter.from_file(chinook_fixture)
    base_conn = adapter.open_readonly()
    migrated_conn = adapter.clone()

    # Migration touches customers, leaving albums untouched
    migration_sql = "ALTER TABLE customers ADD COLUMN loyalty_points INT DEFAULT 0;"
    migrated_conn.executescript(migration_sql)

    inv = evaluate_migration_invariants(
        base_conn,
        migrated_conn,
        target_table="customers",
        elapsed_ms=1.5,
    )

    assert inv.passed is True
    # Verify baseline foreign key check note
    assert any("FOREIGN_KEYS_CLEAN" in c or "Foreign keys clean" in c for c in inv.passed_checks)


def test_rollback_roundtrip_verification(chinook_fixture):
    """Verify up.sql followed by down.sql achieves zero-loss roundtrip."""
    adapter = SQLiteAdapter.from_file(chinook_fixture)
    base_conn = adapter.open_readonly()

    up_sql = "ALTER TABLE customers ADD COLUMN middle_name TEXT DEFAULT '';"
    down_sql = "ALTER TABLE customers DROP COLUMN middle_name;"

    ok, err, lossy = verify_rollback_roundtrip(base_conn, up_sql, down_sql, target_table="customers")
    assert ok is True
    assert err is None
    assert len(lossy) == 0
