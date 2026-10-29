# ⏳ SQL Migration "Dry-Run Time Machine"

> **Autonomous Tool-Using Coding Agent for Safe Database Schema Migrations**  
> Built for the **Nebius x NVIDIA Global AI Hackathon (Coding and Agentic Engineering Track)**.  
> Powered by **NVIDIA Nemotron-3 Super** (120B) & **Nemotron-3 Ultra** (550B) on **Nebius Token Factory**.

[![CI](https://github.com/YaswanthNadh-coder/nebius/actions/workflows/ci.yml/badge.svg)](https://github.com/YaswanthNadh-coder/nebius/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests: 185 passed](https://img.shields.io/badge/tests-185%20passed-brightgreen.svg)](https://github.com/YaswanthNadh-coder/nebius)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 The Problem

Running database migrations in production is terrifying. A single naive `INNER JOIN` against legacy unmapped records or unhandled `NULL` foreign keys silently drops customer records forever. Hand-writing rollback scripts or testing migrations against staging databases with sanitized, toy data regularly fails to catch dirty production edge cases:
- **Silent Row Deletions:** Naive `INNER JOIN`s on nullable or orphan foreign keys silently dropping records.
- **Mononym & String Boundary Breaks:** Single-word names (`"Madonna"`, `"Cher"`) or leading/trailing whitespace (`"\tGrace Hopper\n"`) breaking string splitters.
- **Foreign Key Traps:** Orphan foreign keys (`'LEGACY_99'`) causing runtime crashes when constraints are enforced.
- **Corrupted Numeric Casts:** Currency symbols, negative numbers, and comma separators (`"$120,000"`, `"-$50"`) failing strict integer casts.
- **Lost Schema Artifacts:** Dropped secondary indexes, triggers, and views during SQLite 12-step table rebuilds.
- **Unverified Emergency Rollbacks:** Rollback scripts that crash or permanently lose data when executed.

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

## ⚙️ Installation & Setup Instructions

### Prerequisites
- **Python 3.10+** (tested on Python 3.10, 3.11, 3.12, 3.13)
- **Git** installed on your system
- A **Nebius Token Factory API Key** ([Sign up at Nebius AI Studio](https://tokenfactory.nebius.com/))

---

### Method 1: Local Python Environment Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/YaswanthNadh-coder/nebius.git
cd nebius
```

#### 2. Create and Activate a Virtual Environment

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

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Configure Your Nebius API Key
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Open `.env` and add your Nebius API Key:
```env
NEBIUS_API_KEY=your_actual_nebius_api_key_here
```

> **Note on Key Loading:** The engine automatically strips UTF-8 BOM headers (`\ufeff`), UTF-16 encoding, leading/trailing whitespace, and accidental quotes. You can also paste or override the API key directly in the web app sidebar.

---

### Method 2: Docker Setup

If you prefer running via Docker without configuring a local Python environment:

#### 1. Build the Docker Image
```bash
docker build -t timemachine .
```

#### 2. Run the Container
```bash
# Passing API key via environment variable
docker run -p 8501:8501 -e NEBIUS_API_KEY="your_nebius_api_key_here" timemachine

# Or mounting a local .env file
docker run -p 8501:8501 -v $(pwd)/.env:/app/.env timemachine
```

Open `http://localhost:8501` in your browser.

---

## 🚀 Execution Instructions

### 1. Launch the Web Dashboard (Streamlit)
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to access the interactive 5-tab interface.

---

### 2. Run Headless Migrations via CLI

The Time Machine includes a command-line interface suitable for CI/CD gates and headless automation:

```bash
# Run migration dry-run on a local database file
python -m timemachine run --db path/to/app.sqlite --goal "Split full_name into first_name and last_name" --out migrations/

# Export as Flyway-compatible migration scripts
python -m timemachine run --db app.sqlite --goal goal.txt --out migrations/ --format flyway

# Export as an Alembic migration
python -m timemachine run --db app.sqlite --goal goal.txt --out migrations/ --format alembic
```

#### CLI Exit Codes:
- `0`: **VERIFIED** — Invariants passed and oracle assertions confirmed. Safe to deploy.
- `1`: **FAILED** — Constraint violation, data loss, or execution crash detected.
- `2`: **UNVERIFIED** — Invariant checks passed, but custom transforms require human review.

---

### 3. Run the Test Suite (Offline / Fake LLM)
Run all 185+ unit, integration, and regression tests offline without needing network access or an API key:

```bash
pytest -v
```

Run specific test modules:
```bash
# Verify rollback roundtrip & lossy classification
pytest tests/test_rollback.py

# Verify oscillation detection & sentinel foreign key recipes
pytest tests/test_oscillation_fix.py

# Verify 17 intentional data-mutation traps
pytest tests/test_mutations.py
```

---

### 4. Run the Scale & Performance Benchmark
Test how the Time Machine scales across 10,000, 100,000, and 400,000 row datasets:

```bash
python scripts/bench.py --rows 100000
```

#### Benchmark Targets & Measured Results:
- **100k rows invariants & oracle verification target**: $\le 1.0\text{s}$ (Measured: **~569 ms** — **PASS**)
- **400k rows invariants & oracle verification target**: $\le 3.0\text{s}$ (Measured: **~2.12 s** — **PASS**)
- **Memory footprint**: Flat $O(1)$ memory streaming using `fetchmany(5000)` and SQL aggregate pushdowns.

| Scale Tier | Clone (ms) | Profile (ms) | Migration Exec (ms) | Invariants & Oracle (ms) | Rollback (ms) | Audit Mode |
|---|---|---|---|---|---|---|
| **10,000 rows** | 0.6 ms | 31.34 ms | 6.45 ms | 181.06 ms | 78.06 ms | Exact (Full) |
| **100,000 rows** | 4.91 ms | 231.9 ms | 51.78 ms | 569.27 ms | 1041.41 ms | Sampled (5k+traps) |
| **400,000 rows** | 19.89 ms | 223.87 ms | 239.38 ms | 2120.48 ms | 4508.61 ms | Sampled (5k+traps) |

---

### 5. Run Static Analysis & Linters
```bash
# Run Ruff linter
ruff check timemachine/ tests/

# Run Pyflakes
pyflakes timemachine/ tests/
```

---

## 🖥️ The 5-Step Web Dashboard Workflow

1. **🔌 Tab 1: Connect**
   - Choose a built-in realistic scenario (e.g., *Department Normalization* or *Salary Banding*), upload a `.sqlite`/`.db` file, upload a `.sql` schema dump, or specify a local file path.
   - Inspect table schemas, row counts, secondary indexes, foreign keys, triggers, and views in read-only mode.
   - Configure sandbox limits (size cap, timeouts, foreign key enforcement).

2. **📋 Tab 2: Plan**
   - Provide a natural language migration goal statement.
   - The AI drafts a structured **`MigrationSpec`** containing affected tables, column mappings, transform functions, and null policies.
   - Inspect the generated plan and check the approval box before any migration SQL is drafted.

3. **⚡ Tab 3: Run**
   - Start the autonomous dry-run agent loop.
   - Watch live tool-calling activity (`get_schema`, `profile_table`, `run_on_clone`, `explain`).
   - Observe real-time progress as Nemotron drafts and self-repairs SQL against adversarial traps in an isolated RAM clone.

4. **📊 Tab 4: Result**
   - **Certification Banner:** Clear status badge (`🟢 VERIFIED`, `🟡 UNVERIFIED`, or `🔴 FAILED`).
   - **Forward Migration (`up.sql`):** Transaction-wrapped SQL with pre-checks and the standard 12-step table rebuild.
   - **Reversible Rollback (`down.sql`):** Verified roundtrip script to restore the baseline table.
   - **Rollback Classification:**
     - `✅ Verified` (`EXACT` / `EQUIVALENT_AFTER_NORMALIZATION`): Zero data loss.
     - `⚠️ Lossy`: Warns if the migration goal permanently erased or merged dirty historical data (e.g. orphan codes mapped to `NULL` or `'UNASSIGNED'`).
     - `❌ Failed` (`BROKEN`): The rollback script failed to restore the original table.
   - **Exporters:** One-click downloads for Plain SQL, Flyway migrations, and Alembic versions.

5. **📜 Tab 5: History**
   - Audit trail of previous runs stored in local SQLite (`.timemachine/runs.db`).
   - Review token counts, timings, prompt hashes, and model escalation steps.

---

## 🛡️ Core Verification & Safety Architecture

### 1. Invariant Engine (What IS Verified)
- **Zero Silent Data Loss:** Explicit row accounting distinguishing between deletions (loss) and row duplications.
- **Primary Key Set Invariant:** Exact cryptographic set equality of primary keys before and after migration.
- **Untouched Column Checksums:** Columns declared unchanged must have identical byte checksums.
- **Foreign Key Referential Integrity:** Runs baseline-subtracted `PRAGMA foreign_key_check` under both `FK = ON` and `OFF`.
- **Database Object Preservation:** Secondary indexes, triggers, and views are automatically detected and verified on the rebuilt table.
- **No Leftover Staging Tables:** Ensures temporary tables (`_new`, `_old`, `_tmp`) are safely cleaned up.

### 2. Row-Level Reference Oracles
- Transforms from the reference library (`split_name_first`, `split_name_last`, `strip_honorific`, `cast_int`, `cast_cents`, `default_if_null`, `map_via_lookup`, `normalize_date`, `trim_case`) are verified against independent Python reference functions.
- If all invariants and oracle checks pass, the status is certified **`VERIFIED`**.

### 3. Security Sandbox
- **Authorizer Callback (`set_authorizer`):** Blocks `ATTACH`, `DETACH`, `VACUUM INTO`, `load_extension`, and dangerous administrative PRAGMAs. Only 12 safe inspection PRAGMAs are permitted.
- **Progress Handler Deadline:** Hardware progress handler interrupts execution if an LLM query exceeds the scaled sandbox timeout limit (2.0s–30.0s).
- **Privacy Masking:** Names, emails, and phone numbers are masked into aggregate distributions before entering LLM prompts. Real customer rows are never transmitted.

### 4. Advanced Repair & Escalation Engine
- **2-Step Sentinel Foreign Key Pattern:** Teaches the agent to seed parent tables before inserting orphan records into child tables (`INSERT OR IGNORE INTO parent ...`), preventing foreign key crashes.
- **Automatic Rollback Nullability Relaxation:** Strips unwarranted `NOT NULL` constraints from rollback table definitions if the column was nullable in the baseline schema.
- **Repair Oscillation Detector:** Detects ping-pong loops between `FOREIGN KEY` constraints and `DATA_MISMATCH` errors and provides corrective recipes.
- **Dual-Tier Model Routing:** Starts with fast drafting via **Nemotron-3-Super (120B)** and escalates to **Nemotron-3-Ultra (550B)** when deep multi-step reasoning is required.

---

## 📁 Repository Structure

```
.
├── app.py                     # Streamlit 5-tab web dashboard
├── Dockerfile                 # Production Docker configuration
├── requirements.txt           # Core Python dependencies
├── .env.example               # Environment template for Nebius API key
│
├── timemachine/               # Core Time Machine Engine
│   ├── config.py              # Application settings, timeouts, and env loading
│   ├── adapters/              # Database adapters & introspection (SQLite, PostgreSQL)
│   ├── agent/                 # Autonomous agent loop, prompts, planner, and tools
│   ├── traps/                 # Adversarial edge-case generator and profiler
│   ├── verify/                # Invariant checker, reference oracles, and sandbox
│   ├── export/                # Exporters for Plain SQL, Flyway, and Alembic
│   └── ui/                    # Streamlit UI tabs, components, and views
│
├── scripts/
│   └── bench.py               # 10k, 100k, 400k scale performance benchmark
│
├── evals/
│   ├── demo_chinook.py        # Real-world migration on Chinook database
│   └── run_evals.py           # Evaluation battery against Nebius API
│
└── tests/                     # Test suite (185+ passing tests)
    ├── test_unit.py           # Unit tests (keys, models, sandbox, cleaner)
    ├── test_integration.py    # Integration tests on real SQLite fixtures
    ├── test_mutations.py      # Corpus of 17 intentional migration bugs
    ├── test_agent_loop.py     # Multi-turn agent repair and budget tests
    ├── test_rollback.py       # Rollback verification and classification
    └── test_oscillation_fix.py# Sentinel patterns and oscillation detection
```

---

## 🏗️ Model & Infrastructure Stack

- **AI Inference:** [Nebius Token Factory](https://tokenfactory.nebius.com/)
- **Drafting Model:** `nvidia/nemotron-3-super-120b-a12b` (Low latency)
- **Deep Repair Model:** `nvidia/Nemotron-3-Ultra-550b-a55b` (Advanced reasoning)
- **Database Engine:** `sqlite3` with in-memory `Connection.backup()` RAM cloning
- **SQL Analysis:** `sqlglot` for transactional statement classification and AST validation
- **Web Interface:** `streamlit` with stateful session management

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
