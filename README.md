# ⏳ SQL Migration "Dry-Run Time Machine"

> **Nebius x NVIDIA Global AI Hackathon (Coding and Agentic Engineering Track)**

An autonomous coding agent that eliminates the fear of running destructive SQL schema migrations by combining in-memory SQLite sandboxing with tiered **NVIDIA Nemotron** models hosted on **Nebius Token Factory**.

---

## 🎯 The Problem

Running database migrations in production is terrifying. A single naive `INNER JOIN` against legacy unmapped records or unhandled `NULL` foreign keys can silently drop thousands of customer records forever. Hand-writing rollback scripts or testing migrations against staging databases with stale or sanitized data fails to catch dirty production edge cases.

---

## 💡 The Solution: Dry-Run Time Machine

The SQL Migration Time Machine provides an ephemeral, safe dry-run environment:

1. **Adversarial In-Memory Seeding:** Spins up an ephemeral in-memory SQLite database (`:memory:`) and uses `Faker` to populate 1,000 realistic rows injected with production traps:
   - `NULL` foreign keys
   - Unmapped legacy department codes (`LEGACY_99`)
   - Mononyms and single-word names (`Madonna`)
   - Unicode and apostrophes (`O'Reilly 🚀`)
   - Dollar-prefixed currency strings (`$85000`) and missing salary values
2. **Tiered LLM Routing on Nebius Token Factory:**
   - **Fast Draft (Nemotron 3 Super):** Generates a fast first-pass SQLite migration script.
   - **Self-Healing (Nemotron 3 Ultra):** If verification fails, escalates the failure report and failed SQL to Nemotron 3 Ultra to synthesize a lossless self-healed patch (`LEFT JOIN`, `COALESCE`, safe `INSTR`/`SUBSTR`, `CAST/REPLACE`).
3. **Sub-5ms RAM Snapshots:** Uses `sqlite3.Connection.backup()` to clone the entire seeded database in RAM in `< 5ms`. Each dry-run executes in a completely disposable clone without touching disk or the original state.
4. **Deterministic Mathematical Assertions:**
   - Detects silent row drops (`before_count == after_count`).
   - Audits required target columns (`id`, `first_name`, `last_name`, `dept_code`, `salary_num`, `is_active`).
   - Catches `NULL` corruption in non-nullable target columns.

---

## 📁 Workspace File Structure

```text
.
├── .gitignore          # Git ignore specifications
├── .env.example        # Environment variable template
├── .env                # Your Nebius API key (git-ignored)
├── requirements.txt    # Python dependencies
├── scenarios.py        # Pre-built adversarial SQL migration scenarios
├── agent.py            # Core agent logic: seeding, RAM cloning, verification & self-healing
├── app.py              # Streamlit interactive web dashboard
├── test_agent.py       # CLI smoke test script
└── README.md           # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Your Nebius API Key

In `.env`, paste your Nebius Token Factory API key:

```env
NEBIUS_API_KEY=your_actual_nebius_api_key_here
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

### 5. (Optional) Run the CLI Smoke Test

```bash
python test_agent.py
```

---

## 🏗️ Architecture & Technologies

- **Nebius Token Factory:** Serverless OpenAI-compatible inference endpoint (`https://api.tokenfactory.nebius.com/v1/`).
- **NVIDIA Nemotron 3 Super & Ultra:** Tiered fast drafting and deep agentic self-healing.
- **SQLite In-Memory (`:memory:`):** Fast, isolated, zero-disk database sandboxing.
- **Faker:** Synthetic dirty data generation simulating real-world production data.
- **Streamlit:** Interactive visual timeline displaying attempts, metrics, failure telemetry, and verified output rows.
