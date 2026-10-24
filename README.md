# ⏳ SQL Migration "Dry-Run Time Machine"

> **Nebius × NVIDIA Global AI Hackathon — Coding & Agentic Engineering Track**

An autonomous coding agent that eliminates the fear of running destructive SQL schema migrations. It drafts, tests, and self-heals migration scripts entirely in RAM — so your production database is never touched.

---

## 🧠 How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│  1. SEED        Faker populates 1,000 adversarial rows in RAM   │
│  2. DRAFT       Nemotron 3 Super generates initial SQL migration │
│  3. CLONE       sqlite3.backup() snapshots the DB in <5 ms      │
│  4. VERIFY      Deterministic assertions catch silent data loss  │
│  5. SELF-HEAL   Nemotron 3 Ultra rewrites lossless SQL if needed │
│  6. REPEAT      Fresh clone → re-verify until 1,000/1,000 pass  │
└──────────────────────────────────────────────────────────────────┘
```

### Tiered Model Routing (Nebius Token Factory)

| Stage | Model | Why |
|-------|-------|-----|
| Fast Draft | **NVIDIA Nemotron 3 Super** | Low-latency first-pass SQL generation |
| Self-Heal | **NVIDIA Nemotron 3 Ultra** | Deep reasoning over failure telemetry to synthesize lossless patches |

Both models are accessed through the **Nebius Token Factory** API (`https://api.tokenfactory.nebius.com/v1/`), which provides serverless inference with zero infrastructure overhead.

### Sub-5 ms RAM Cloning

Instead of writing to disk, the agent uses **`sqlite3.Connection.backup()`** to clone the entire in-memory database in under 5 milliseconds. Each verification attempt runs against a disposable clone, so the original seeded snapshot is never mutated.

### Adversarial Edge Cases

The synthetic dataset is deliberately injected with production traps that break naive migrations:

- **NULL foreign keys** — `dept_code IS NULL` causes `INNER JOIN` to silently drop rows
- **Unmapped legacy codes** — `'LEGACY_99'` has no match in `departments`
- **Single-word names** — `'Madonna'` breaks `SUBSTR`/`INSTR` splitting on space
- **Unicode & apostrophes** — `"O'Reilly 🚀"` tests string-handling robustness
- **Dollar-prefixed salary strings** — `'$85000'` and `NULL` test `CAST`/`REPLACE` pipelines

---

## 🚀 Quick Start

### 1. Clone & Enter the Project

```bash
cd nebius
```

### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your API Key

Edit the `.env` file and replace the placeholder with your real Nebius Token Factory key:

```env
NEBIUS_API_KEY=your_real_key_here
```

### 5. Launch the App

```bash
streamlit run app.py
```

The Streamlit UI will open in your browser. Select a scenario, click **"🚀 Run Adversarial Time-Machine Dry Run"**, and watch the agent draft → fail → self-heal in real time.

### 6. (Optional) CLI Smoke Test

```bash
python test_agent.py
```

---

## 📁 Project Structure

```
.
├── .gitignore          # Git ignore rules
├── .env.example        # Template for required environment variables
├── .env                # Your actual API key (git-ignored)
├── requirements.txt    # Python dependencies
├── scenarios.py        # Pre-built adversarial migration scenarios
├── agent.py            # Core agent: seeder, cloner, verifier, self-heal loop
├── app.py              # Streamlit UI
├── test_agent.py       # CLI smoke test
└── README.md           # This file
```

---

## 🔑 Key Technical Decisions

1. **In-Memory Only** — Zero disk I/O. The entire pipeline runs in RAM using `sqlite3.connect(":memory:")`, making it safe and fast.
2. **Deterministic Assertions** — No LLM is used to judge pass/fail. Row counts, column presence, and NULL checks are pure mathematical comparisons.
3. **Tiered Cost Optimization** — The cheaper/faster Nemotron Super handles the optimistic first draft. The more expensive Nemotron Ultra is only invoked when the sandbox catches a real failure.
4. **Automatic Model Discovery** — `resolve_nebius_models()` queries the Nebius API to detect the exact model IDs available on your account, with graceful fallback to defaults.

---

## 🏗️ Built With

- [Nebius Token Factory](https://nebius.com/) — Serverless NVIDIA GPU inference
- [NVIDIA Nemotron 3 Super & Ultra](https://build.nvidia.com/) — Tiered LLM reasoning
- [Streamlit](https://streamlit.io/) — Interactive web UI
- [Faker](https://faker.readthedocs.io/) — Realistic synthetic data generation
- [SQLite3](https://docs.python.org/3/library/sqlite3.html) — In-memory database with `backup()` cloning

---

## 📄 License

MIT — built for the Nebius × NVIDIA Global AI Hackathon.
