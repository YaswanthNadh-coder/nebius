try:
    import sqlglot
    from sqlglot import exp
    HAS_SQLGLOT = True
except ImportError:
    sqlglot = None  # type: ignore[assignment]
    exp = None  # type: ignore[assignment]
    HAS_SQLGLOT = False

from timemachine.logging import get_logger

logger = get_logger("dialect")


def classify_sql(sql: str, dialect: str = "sqlite") -> list[str]:
    """Return statement types present in the SQL script."""
    statement_types: list[str] = []
    if not HAS_SQLGLOT:
        for line in sql.split(";"):
            cleaned = line.strip().upper()
            if cleaned:
                statement_types.append(cleaned.split()[0])
        return statement_types

    try:
        parsed = sqlglot.parse(sql, read=dialect)
        for stmt in parsed:
            if stmt is None:
                continue
            statement_types.append(stmt.key.upper())
    except Exception as exc:
        logger.debug("sqlglot parsing notice: %s", exc)
    return statement_types


def detect_dangerous_operations(sql: str, dialect: str = "sqlite") -> list[str]:
    """Detect risky operations: DROP TABLE, DROP COLUMN, TRUNCATE, DELETE without WHERE."""
    risks: list[str] = []
    if not HAS_SQLGLOT:
        lower = sql.lower()
        if "drop table" in lower:
            risks.append("DROP TABLE statement")
        if "drop column" in lower:
            risks.append("DROP COLUMN statement")
        if "delete from" in lower and "where" not in lower:
            risks.append("DELETE without WHERE")
        if "truncate table" in lower or "truncate " in lower:
            risks.append("TRUNCATE TABLE statement")
        return list(dict.fromkeys(risks))

    try:
        parsed = sqlglot.parse(sql, read=dialect)
        for stmt in parsed:
            if stmt is None:
                continue

            # DROP TABLE
            if isinstance(stmt, exp.Drop):
                kind = stmt.args.get("kind", "TABLE").upper()
                target = stmt.this.sql(dialect=dialect) if stmt.this else ""
                risks.append(f"DROP {kind} {target}".strip())

            # ALTER TABLE ... DROP COLUMN
            if isinstance(stmt, exp.Alter):
                for action in stmt.args.get("actions", []):
                    if isinstance(action, exp.Drop) or "drop" in action.sql().lower():
                        risks.append(f"ALTER TABLE DROP: {action.sql(dialect=dialect)}")

            # DELETE without WHERE
            if isinstance(stmt, exp.Delete):
                if not stmt.args.get("where"):
                    tbl = stmt.this.sql(dialect=dialect) if stmt.this else "table"
                    risks.append(f"DELETE FROM {tbl} without WHERE clause (deletes all rows)")

            # TRUNCATE
            if stmt.key.upper() == "TRUNCATE":
                risks.append("TRUNCATE TABLE statement")

    except Exception as exc:
        logger.debug("sqlglot risk inspection fallback: %s", exc)
        # Fallback text heuristics
        lower = sql.lower()
        if "drop table" in lower:
            risks.append("DROP TABLE statement")
        if "drop column" in lower:
            risks.append("DROP COLUMN statement")
        if "delete from" in lower and "where" not in lower:
            risks.append("DELETE without WHERE")
        if "truncate table" in lower or "truncate " in lower:
            risks.append("TRUNCATE TABLE statement")

    return list(dict.fromkeys(risks))
