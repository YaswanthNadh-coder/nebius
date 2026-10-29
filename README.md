# ⏳ SQL Migration "Dry-Run Time Machine"

> **Autonomous Tool-Using Coding Agent for Safe Database Schema Migrations**  
> Built for the **Nebius x NVIDIA Global AI Hackathon (Coding and Agentic Engineering Track)**.  
> Powered by **NVIDIA Nemotron-3 Super** & **Nemotron-3 Ultra** on **Nebius Token Factory**.

[![CI](https://github.com/YaswanthNadh-coder/nebius/actions/workflows/ci.yml/badge.svg)](https://github.com/YaswanthNadh-coder/nebius/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 The Problem

Running database migrations in production is terrifying. A single naive `INNER JOIN` against legacy unmapped records or unhandled `NULL` foreign keys silently drops customer records forever. Hand-writing rollback scripts or testing migrations against staging databases with sanitized, toy data regularly fails to catch dirty production edge cases:
- Hidden single-word mononyms (`"Madonna"`, `"Prince"`) breaking naive string splitters.
- Unhandled currency strings (`"$85,000"`, `"-$500"`) corrupting numeric casts.
- Orphan foreign keys causing silent data loss under `INNER JOIN`.
- Dropped views, triggers, and secondary indexes when recreating SQLite tables without the documented 12-step procedure.

---

## 💡 The Solution: Spec-First Dry-Run Time Machine

The **SQL Migration Time Machine** is an autonomous, tool-using agent that connects to your database, proves a migration is 100% safe on a disposable in-RAM clone, and delivers verified forward migrations, reversible down rollbacks, and comprehensive audit reports.

**Core Safety Invariant:** Generated SQL **NEVER** touches or executes on your source database. All introspection connects via read-only URI mode (`file:...mode=ro`), and all dry-run execution happens inside isolated in-memory clones with cryptographic byte verification.

```
┌─────────────────┐       Read-Only        ┌────────────────────────┐
│ User Database   │ ─────────────────────> │ Deep Introspection     │
│ (SQLite / File) │   (Zero-Writes ro)     │ Schema + Column Stats  │
└─────────────────┘                        └──────────┬─────────────┘
                                                      │
                                                      ▼
┌───────────────────────────────────────────────────────────────────┐
│ Plan: MigrationSpec (Pydantic Schema)                             │
│ Affected Tables, Column Mappings, Transforms, Expected Constraints│
│ ──> Displayed for Human Review & Explicit Approval                │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ Autonomous Tool-Using Agent Loop                                  │
│   • Model Drafting: NVIDIA Nemotron-3-Super (120b)                │
│   • Model Escalation: NVIDIA Nemotron-3-Ultra (550b)              │
│   • Tools: get_schema, profile_table, run_on_clone, explain       │
│   • In-Memory Sandbox Clone (High-Speed RAM clone, Authorizer + 2s Deadline) │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ Spec-First Verification & Oracles                                 │
│   • Baseline-Subtracted Invariants (FKs, indexes, views, triggers)│
│   • Row-Level Oracle Engine (Independent Python Reference Transforms)
│   • Reversible Down Rollback Round-Trip Check                     │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ Result Artifacts: VERIFIED / UNVERIFIED / FAILED                  │
│   • Transactional up.sql (with pre-checks) & down.sql             │
│   • Markdown Audit Report & Trap Breakdown Matrix                 │
│   • Exporters: Plain SQL, Flyway, Alembic                         │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🛡️ Verification Engine: What Is and Is NOT Verified

The Time Machine enforces clear, deterministic boundaries between what is proven correct vs. what requires human review:

### What IS Verified:
1. **Deterministic Invariant Checks:**
   - **Row Accounting:** Explicit distinction between row *loss* (e.g. naive `INNER JOIN`) and row *duplication* (e.g. bad cross joins).
   - **Primary Key Preservation:** Exact set identity of primary keys before and after migration.
   - **Untouched Column Checksums:** Per-row cryptographic checksum of columns declared unchanged.
   - **Schema Constraints:** Validation of `NOT NULL`, defaults, primary keys, and types via `PRAGMA table_info`.
   - **Database Objects Preservation:** All non-target tables, secondary indexes, triggers, and views that existed before migration must exist and be valid afterward.
   - **Baseline-Subtracted Foreign Key Integrity:** Runs `PRAGMA foreign_key_check` and `PRAGMA integrity_check` before and after; pre-existing source violations are reported as baseline defects, not blamed on the migration.
   - **Foreign Key Dual-Mode:** Every migration is tested under both `PRAGMA foreign_keys = ON` and `OFF`.
   - **No Leftover Temp Tables:** Flags any leftover `_new`, `_tmp`, or staging tables.
   - **Rollback Round-Trip:** Verifies `down.sql` by applying `up.sql` then `down.sql` on a clone and asserting table checksum identity.

2. **Row-Level Reference Oracles (`VERIFIED` status):**
   - For standard transformations (`split_name`, `strip_honorific`, `cast_int`, `cast_cents`, `default_if_null`, `map_via_lookup`, `normalize_date`, `trim`, `lower`, `upper`), an independent Python reference implementation runs alongside the SQL. Every migrated row is compared against the oracle. When invariants and oracle agree, the run is marked **`VERIFIED`**.

### What is NOT Verified (`UNVERIFIED` status):
- For complex, proprietary business transformations outside the reference library where no independent deterministic oracle is configured, the migration will be marked **`UNVERIFIED`** (provided all invariant checks pass).
- The system will **NEVER** falsely claim "VERIFIED" for unverified custom logic. It explicitly generates an audit warning and highlights the rows requiring human sign-off.

---

## 🔒 Security Sandbox

Untrusted or LLM-generated SQL is executed strictly inside a secured SQLite environment:
- **`set_authorizer`:** Denies `ATTACH`, `DETACH`, `VACUUM INTO`, `load_extension`, and any `PRAGMA` other than read-only schema inspection (`table_info`, `foreign_key_check`, `integrity_check`, `index_list`, `foreign_keys`).
- **`set_progress_handler`:** Enforces a hard **2.0-second execution deadline** preventing infinite recursion, runaway Cartesian products, and locking loops.
- **Identifier Quoting:** All table and column identifiers are validated and quoted.
- **Privacy Masking:** Column profiler masks emails, phone numbers, and names before passing aggregate statistics to the LLM. Real data rows are never sent into prompt templates.

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.10+** (tested on Python 3.10, 3.11, 3.12, 3.13)
- Nebius Token Factory API Key ([Nebius AI Studio](https://tokenfactory.nebius.com/))

### 1. Clone the Repository
```bash
git clone https://github.com/YaswanthNadh-coder/nebius.git
cd nebius
```

### 2. Create and Activate Virtual Environment
**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your Nebius API Key
Create a `.env` file in the project root (or copy `.env.example`):
```bash
cp .env.example .env
```
In `.env`, set your key:
```env
NEBIUS_API_KEY=your_nebius_api_key_here
```
> **Robust Key Loading:** The application automatically handles UTF-8 BOM (`\ufeff`), UTF-16, surrounding whitespace, and quotes. You can also provide or override the key directly in the Streamlit sidebar.
>
> ⚠️ **Security Note:** Never paste API keys into public chats, issues, or pull requests, and never commit `.env` to source control. The repository `.gitignore` ensures your `.env` remains local and ignored.

### 5. Launch the Streamlit Web Dashboard
```bash
streamlit run app.py
```

### 6. (Optional) Run CLI Smoke Test
```bash
python legacy/test_agent.py
```
> **Note:** Legacy single-file scripts (`agent.py`, `scenarios.py`, `test_agent.py`) have been moved to `legacy/`. The production system runs via `streamlit run app.py` and the `timemachine/` package.

---

## 🖥️ Streamlit 5-Tab Interface

The web interface is organized into five persistent tabs (`st.session_state` preserved across interactions):

1. **🔌 Connect:**
   - Upload a `.sqlite`/`.db` file, provide a local path, paste a `.sql` schema dump, or pick one of the built-in realistic demo scenarios.
   - **Upload State Persistence:** Uploaded files and schema dumps are fingerprinted `(name, size, sha256)` and loaded into memory once per file. Subsequent widget interactions and Streamlit reruns preserve your generated MigrationSpec and approval state without resetting or reloading the file.
   - Deep schema explorer inspecting tables, columns, indexes, foreign keys, triggers, and views.
   - Configure size caps (default 200 MB) and foreign key enforcement toggle.
2. **📋 Plan:**
   - Define the migration goal in natural language.
   - Generates a structured `MigrationSpec` showing affected tables, column mappings, transforms, and null policies.
   - Editable form requiring explicit user approval before any SQL is drafted.
3. **⚡ Run:**
   - Live step-by-step agent trace with `st.status`.
   - Tool calling activity: schema introspection, table profiling, dry-run clone audits.
   - Live before/after sample row diffs and adversarial trap results.
4. **📊 Result:**
   - Final certification banner (`VERIFIED`, `UNVERIFIED`, or `FAILED`).
   - Transactional `up.sql` with safety pre-checks and reversible `down.sql`.
   - Formatted Markdown audit report and invariant matrix.
   - One-click downloads and exports for Plain SQL, Flyway, and Alembic.
5. **📜 History:**
   - Complete local SQLite run audit log (`.timemachine/runs.db`) detailing past migrations, prompt hashes, models, tokens, and timings.

---

## 💻 Command Line Interface (CLI)

The Time Machine can gate CI/CD pipelines and run headlessly:

```bash
# Run migration dry-run on a database file
python -m timemachine run --db app.sqlite --goal "Split full_name into first_name and last_name" --out migrations/

# Export as Flyway migration
python -m timemachine run --db app.sqlite --goal goal.txt --out migrations/ --format flyway

# Export as Alembic migration
python -m timemachine run --db app.sqlite --goal goal.txt --out migrations/ --format alembic
```

### CLI Exit Codes
- **`0`**: **VERIFIED** — Invariants passed and oracle assertions confirmed. Safe to deploy.
- **`1`**: **FAILED** — Constraint violation, data loss, or execution error detected.
- **`2`**: **UNVERIFIED** — Invariant checks passed, but custom transforms require human review.

### Agent Stop Reasons
- **`SUCCESS`**: All invariant and oracle verification checks passed (`VERIFIED` or `UNVERIFIED`).
- **`NO_PROGRESS`**: Model returned identical SQL across repair attempts without making progress.
- **`SPEC_INVALID`**: MigrationSpec failed schema validation checks against the live database schema (e.g. unknown target table, missing source column).
- **`PLAN_FAILED`**: Model failed to generate a valid MigrationSpec conforming to JSONSchema.
- **`MODEL_RETURNED_NO_SQL`**: Model produced consecutive responses containing no executable SQL statements.
- **`STEP_BUDGET_EXHAUSTED`**: Step budget exceeded without satisfying all verification invariants.
- **`WALL_CLOCK_TIMEOUT`**: Total agent execution time exceeded the configured wall-clock budget (default 180s).

---

## 🧪 Testing & Evaluation

### Offline Unit & Integration Tests (Fake LLM, Zero Network)
Run the full 51-test suite locally without requiring an API key or network access:
```bash
pytest -q
```
Includes:
- **Unit Tests (`test_unit.py`):** Key loading (BOM, UTF-16, quotes, placeholders), model resolution, sandbox authorizer, deadline handler, SQL cleaner, oracle transforms.
- **Integration Tests (`test_integration.py`):** Real SQLite fixture with indexes, views, triggers, and foreign keys tested under both FK ON and OFF modes.
- **Mutation Corpus (`test_mutations.py`):** 17 intentional defect scenarios (row loss, duplicate rows, broken FKs, dropped indexes, dropped views/triggers, type coercion, silent truncations, leftover temp tables, sandbox escapes) — every single one is caught and flagged.
- **Agent Loop Tests (`test_agent_loop.py`):** Multi-step repair loops, model escalation, budget exhaustion, error recovery.
- **Streamlit App Tests (`test_app.py`):** Session state persistence and error handling.

### Chinook Real-Database Demo
Execute real migrations against a cloned Chinook database (tracks, artists, albums, customers) with byte-identical source verification:
```bash
python evals/demo_chinook.py
```

### Live Nebius Evaluation Suite (Requires Valid API Key)
Benchmarks Nemotron Super vs. Tiered Routing vs. Nemotron Ultra across 13 real-world migration tasks:
```bash
python evals/run_evals.py
```

### Scale & Verification Performance Benchmark (`scripts/bench.py`)
Run the scaling benchmark across 10k, 100k, and 400k row scale tiers:
```bash
python scripts/bench.py
```

#### Targets & Performance Results:
- **100k rows invariants & oracle verification target**: $\le 1.0\text{s}$ (Measured: **~574 ms** — **PASS**)
- **400k rows invariants & oracle verification target**: $\le 3.0\text{s}$ (Measured: **~1.90 s** — **PASS**)
- **Memory footprint**: Flat $O(1)$ memory streaming via `fetchmany(5000)` and SQL aggregate pushdowns (above 20k rows, audits all trap rows, a deterministic 5k sample, and aggregate checksums).

| Scale Tier | Clone (ms) | Profile (ms) | Migration Exec (ms) | Invariants & Oracle (ms) | Rollback (ms) | Audit Mode |
|---|---|---|---|---|---|---|
| **10,000 rows** | 0.41 ms | 30.71 ms | 4.32 ms | 157.48 ms | 77.06 ms | Exact (Full) |
| **100,000 rows** | 4.07 ms | 195.73 ms | 53.26 ms | 574.07 ms | 854.95 ms | Sampled (5k+traps) |
| **400,000 rows** | 14.59 ms | 198.12 ms | 216.57 ms | 1897.62 ms | 3720.14 ms | Sampled (5k+traps) |

---

## 🏗️ Architecture & Model Routing

- **Inference Provider:** [Nebius Token Factory](https://tokenfactory.nebius.com/) (`https://api.tokenfactory.nebius.com/v1/`)
- **Fast Drafting Model:** `nvidia/nemotron-3-super-120b-a12b` ("Super") — Ultra-low latency draft generation.
- **Self-Healing Reasoning Model:** `nvidia/Nemotron-3-Ultra-550b-a55b` ("Ultra") — Deep multi-step repair when audits flag constraint violations or data loss.
- **Database Engine:** `sqlite3` with high-speed in-memory `Connection.backup()` RAM cloning (fast backup and restore times tracked separately).
- **Dialect Analysis:** `sqlglot` for transactional statement classification and dangerous operation linting.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
