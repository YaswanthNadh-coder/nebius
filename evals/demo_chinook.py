"""Chinook demo script executing real migrations on Chinook SQLite database."""

import hashlib
import os
import sqlite3
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows-safe stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from timemachine.adapters.sqlite import SQLiteAdapter
from timemachine.agent.loop import run_time_machine_agent
from timemachine.config import get_api_key_info, is_placeholder_key, load_env_file
from timemachine.verify.spec import ColumnMapping, MigrationSpec


def create_demo_chinook_db(db_path: str) -> None:
    """Create local Chinook demo SQLite database fresh."""
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

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
        CREATE TABLE tracks (
            TrackId INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL,
            AlbumId INTEGER,
            UnitPrice REAL NOT NULL,
            FOREIGN KEY (AlbumId) REFERENCES albums(AlbumId)
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

    # Indexes, views, triggers
    cur.execute("CREATE INDEX idx_track_album ON tracks(AlbumId);")
    cur.execute("CREATE UNIQUE INDEX idx_customer_phone ON customers(Phone);")
    cur.execute("CREATE VIEW v_album_titles AS SELECT a.AlbumId, a.Title, art.Name FROM albums a JOIN artists art ON a.ArtistId = art.ArtistId;")
    cur.execute("CREATE TRIGGER trg_track_audit AFTER INSERT ON tracks BEGIN SELECT 1; END;")

    # Seed rows
    cur.executemany("INSERT INTO artists VALUES (?, ?);", [(1, "Queen"), (2, "Pink Floyd")])
    cur.executemany("INSERT INTO albums VALUES (?, ?, ?);", [(1, "A Night at the Opera", 1), (2, "The Wall", 2)])
    cur.executemany(
        "INSERT INTO tracks VALUES (?, ?, ?, ?);",
        [(1, "Bohemian Rhapsody", 1, 1.99), (2, "Comfortably Numb", 2, 0.99)],
    )
    cur.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?);",
        [
            (1, "Freddie", "Mercury", "EMI", "UK", "+44 20 7946 0991"),
            (2, "David", "Gilmour", None, "UK", "+44 20 7946 0992"),
            (3, "Roger", "Waters", None, "USA", "+1 212 555 0199"),
        ],
    )
    conn.commit()
    conn.close()


def run_chinook_demo():
    """Run all 4 Chinook migration demo goals on a real SQLite database."""
    load_env_file()
    db_file = "chinook_demo.db"
    create_demo_chinook_db(db_file)

    # Compute source file hash before run
    with open(db_file, "rb") as f:
        sha_initial = hashlib.sha256(f.read()).hexdigest()

    adapter = SQLiteAdapter.from_file(db_file)

    demo_goals = [
        {
            "name": "1. NOT NULL Company with default 'N/A'",
            "table": "customers",
            "sql": """
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
            """,
            "spec": MigrationSpec(
                affected_tables=["customers"],
                column_mappings={
                    "customers": [
                        ColumnMapping(new_column="Company", source_column="Company", transform="default_if_null", default_value="N/A"),
                    ]
                },
                row_policy="preserve_all",
                columns_declared_unchanged={"customers": ["CustomerId", "FirstName", "LastName", "Country", "Phone"]},
            ),
        },
        {
            "name": "2. UnitPrice to integer cents",
            "table": "tracks",
            "sql": """
                ALTER TABLE tracks ADD COLUMN unit_price_cents INTEGER NOT NULL DEFAULT 0;
                UPDATE tracks SET unit_price_cents = CAST(ROUND(UnitPrice * 100) AS INTEGER);
            """,
            "spec": MigrationSpec(
                affected_tables=["tracks"],
                column_mappings={
                    "tracks": [
                        ColumnMapping(new_column="unit_price_cents", source_column="UnitPrice", transform="cast_cents", default_value=0),
                    ]
                },
                row_policy="preserve_all",
                columns_declared_unchanged={"tracks": ["TrackId", "Name", "AlbumId", "UnitPrice"]},
            ),
        },
        {
            "name": "3. Split Phone into country code and local number",
            "table": "customers",
            "sql": """
                ALTER TABLE customers ADD COLUMN phone_country_code TEXT DEFAULT '+1';
                ALTER TABLE customers ADD COLUMN local_phone TEXT DEFAULT '';
                UPDATE customers SET
                    phone_country_code = CASE WHEN INSTR(Phone, ' ') > 0 THEN SUBSTR(Phone, 1, INSTR(Phone, ' ') - 1) ELSE '+1' END,
                    local_phone = CASE WHEN INSTR(Phone, ' ') > 0 THEN SUBSTR(Phone, INSTR(Phone, ' ') + 1) ELSE Phone END;
            """,
            "spec": MigrationSpec(
                affected_tables=["customers"],
                row_policy="preserve_all",
                columns_declared_unchanged={"customers": ["CustomerId", "FirstName", "LastName", "Company", "Country", "Phone"]},
            ),
        },
        {
            "name": "4. Extract Country lookup table with FK",
            "table": "customers",
            "sql": """
                CREATE TABLE IF NOT EXISTS countries (
                    CountryId INTEGER PRIMARY KEY AUTOINCREMENT,
                    Name TEXT UNIQUE NOT NULL
                );
                INSERT OR IGNORE INTO countries (Name) SELECT DISTINCT Country FROM customers WHERE Country IS NOT NULL;
                ALTER TABLE customers ADD COLUMN CountryId INTEGER REFERENCES countries(CountryId);
                UPDATE customers SET CountryId = (SELECT CountryId FROM countries WHERE countries.Name = customers.Country);
            """,
            "spec": MigrationSpec(
                affected_tables=["customers", "countries"],
                row_policy="preserve_all",
                columns_declared_unchanged={"customers": ["CustomerId", "FirstName", "LastName", "Company", "Country", "Phone"]},
            ),
        },
    ]

    print("=" * 70)
    print("RUNNING CHINOOK DEMO MIGRATIONS ON DISPOSABLE RAM CLONES")
    print(f"Source DB: {db_file} (SHA256: {sha_initial[:12]}...)")
    print("=" * 70)

    for g in demo_goals:
        print(f"\nDemo Migration: {g['name']} (Table: {g['table']})")
        # Run on clone under FK=ON
        c_on = adapter.clone()
        ok_on, err_on, elapsed_on = adapter.run_in_sandbox(c_on, g["sql"], fk_enabled=True)
        assert ok_on is True, f"Failed under FK ON: {err_on}"

        # Run on clone under FK=OFF
        c_off = adapter.clone()
        ok_off, err_off, elapsed_off = adapter.run_in_sandbox(c_off, g["sql"], fk_enabled=False)
        assert ok_off is True, f"Failed under FK OFF: {err_off}"

        # Evaluate deep invariants on clone
        from timemachine.verify.invariants import evaluate_migration_invariants
        base = adapter.open_readonly()
        inv = evaluate_migration_invariants(
            base,
            c_on,
            target_table=g["table"],
            spec=g["spec"],
            elapsed_ms=elapsed_on,
        )
        print(f"  -> Outcome: {inv.status} (Invariants passed: {len(inv.passed_checks)}, failed: {len(inv.failed_checks)})")
        assert inv.status in ("VERIFIED", "UNVERIFIED"), f"Expected VERIFIED/UNVERIFIED, got {inv.status}: {inv.failed_checks}"

        c_on.close()
        c_off.close()

    # Verify source database file is 100% byte-identical
    with open(db_file, "rb") as f:
        sha_final = hashlib.sha256(f.read()).hexdigest()

    assert sha_initial == sha_final, "Source database file was modified!"
    print("\n" + "=" * 70)
    print("ALL CHINOOK DEMOS PASSED!")
    print(f"Verified: Source DB is byte-identical (SHA256: {sha_final[:12]}...)")
    print("=" * 70)

    # Close adapter connection before file removal
    adapter.close()

    # Clean up demo file
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except OSError:
            pass


if __name__ == "__main__":
    run_chinook_demo()
