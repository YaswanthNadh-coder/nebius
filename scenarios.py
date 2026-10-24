"""Pre-built adversarial SQL migration scenarios for live hackathon demos."""

SCENARIOS = {
    "Scenario 1: Employee Table Rebuild (NULL & Unmatched Foreign Key Trap)": {
        "schema": """CREATE TABLE departments (
    code TEXT PRIMARY KEY,
    dept_name TEXT
);
INSERT INTO departments VALUES ('ENG', 'Engineering'), ('FIN', 'Finance'), ('OPS', 'Operations');

CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    full_name TEXT,
    dept_code TEXT,
    salary_text TEXT,
    is_active INTEGER
);""",
        "goal": """Rebuild the `employees` table so that:
1. `full_name` is split into non-null `first_name` and `last_name` TEXT columns.
2. `salary_text` (like '$85000' or NULL) is converted into a non-null INTEGER column `salary_num` (defaulting to 0).
3. `dept_code` is validated against `departments` with zero NULLs allowed (default missing or unmatched legacy codes to 'UNASSIGNED').
4. Preserve all 1,000 rows and keep `id` and `is_active` (defaulting NULL `is_active` to 0).""",
        "required_cols": ["id", "first_name", "last_name", "dept_code", "salary_num", "is_active"],
        "non_null_cols": ["first_name", "last_name", "dept_code", "salary_num", "is_active"],
    },
    "Scenario 2: Payroll Currency Normalization & Bonus Consolidation": {
        "schema": """CREATE TABLE departments (
    code TEXT PRIMARY KEY,
    dept_name TEXT
);
INSERT INTO departments VALUES ('ENG', 'Engineering'), ('FIN', 'Finance'), ('OPS', 'Operations');

CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    full_name TEXT,
    dept_code TEXT,
    salary_text TEXT,
    is_active INTEGER
);""",
        "goal": """Transform `employees` into a clean payroll schema while keeping the table name `employees`:
1. Split `full_name` into non-null `first_name` and `last_name` (if single-word name, set `last_name` to 'N/A').
2. Parse `salary_text` ('$120000' or NULL) into integer `salary_num` (default 0).
3. Normalize `dept_code` against `departments` so any NULL or unmapped code becomes 'UNASSIGNED'.
4. Ensure `is_active` has no NULL values (COALESCE to 0).""",
        "required_cols": ["id", "first_name", "last_name", "dept_code", "salary_num", "is_active"],
        "non_null_cols": ["first_name", "last_name", "dept_code", "salary_num", "is_active"],
    },
}
