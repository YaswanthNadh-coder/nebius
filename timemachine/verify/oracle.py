"""Independent row-level oracle reference transforms and verification engine."""

import re
from datetime import datetime
from typing import Any

from timemachine.logging import get_logger
from timemachine.verify.spec import ColumnMapping

logger = get_logger("verify.oracle")

_HONORIFICS = re.compile(
    r"^(mr\.?|mrs\.?|ms\.?|dr\.?|prof\.?|rev\.?|hon\.?)\s+",
    re.IGNORECASE,
)
_SUFFIXES = re.compile(
    r",?\s+(jr\.?|sr\.?|ii|iii|iv|md|phd|dds|esq\.?)$",
    re.IGNORECASE,
)


def passthrough(val: Any) -> Any:
    """Preserve source column value without alteration."""
    return val


def strip_honorific(val: Any) -> str:
    """Strip title prefixes and credential suffixes from name string after stripping all whitespace with Python str.strip()."""
    if val is None:
        return ""
    s = str(val).strip()
    s = _HONORIFICS.sub("", s).strip()
    s = _SUFFIXES.sub("", s).strip()
    return s


def split_name_first(val: Any) -> str:
    """Strip ALL whitespace (including tab/newline with Python str.strip()), then take text before the first space."""
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    if " " in s:
        return s.split(" ", 1)[0].strip()
    return s


def split_name_last(val: Any, default: str = "N/A") -> str:
    """Remainder after the first space (stripped of all whitespace with Python str.strip()), or the default 'N/A' for single-word names."""
    if val is None:
        return default
    s = str(val).strip()
    if not s:
        return default
    if " " in s:
        remainder = s.split(" ", 1)[1].strip()
        return remainder if remainder else default
    return default


def cast_int(val: Any, default: int = 0) -> int:
    """Parse integer, safely stripping currency symbols ($/€/£), commas, and all whitespace with Python str.strip()."""
    if val is None:
        return default
    s = str(val).strip()
    is_negative = False
    if s.startswith("-"):
        is_negative = True
        s = s[1:]
    s = s.replace("$", "").replace("€", "").replace("£", "").replace(",", "").replace(" ", "").strip()
    if not s:
        return default
    try:
        if "." in s:
            num = int(float(s))
        else:
            num = int(s)
        return -num if is_negative else num
    except (ValueError, TypeError):
        return default


def cast_cents(val: Any, default: int = 0) -> int:
    """Convert currency float/string to integer cents (e.g. 14.99 -> 1499), stripping currency symbols and all whitespace with Python str.strip()."""
    if val is None:
        return default
    s = str(val).strip().replace("$", "").replace(",", "").replace(" ", "")
    if not s:
        return default
    try:
        f = float(s)
        return int(round(f * 100))
    except (ValueError, TypeError):
        return default


def default_if_null(val: Any, default: Any = "") -> Any:
    """Coalesce None/empty values to default."""
    if val is None:
        return default
    if isinstance(val, str) and not val.strip():
        return default
    return val


def map_via_lookup(val: Any, lookup_dict: dict[str, str], default: Any = "UNASSIGNED") -> Any:
    """Map foreign key or code via dictionary, falling back to default."""
    if val is None:
        return default
    s = str(val).strip()
    return lookup_dict.get(s, default)


def normalize_date(val: Any) -> str:
    """Normalize date string to ISO YYYY-MM-DD."""
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""

    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s


def trim_case(val: Any, mode: str = "trim") -> str:
    """Apply trim, lower, or upper string transformations after stripping all whitespace with Python str.strip()."""
    if val is None:
        return ""
    s = str(val).strip()
    if mode == "lower":
        return s.lower()
    if mode == "upper":
        return s.upper()
    return s


def salary_band(val: Any) -> str:
    """Categorize integer salary into bands: 'HIGH' for >= 100000, 'MID' for >= 60000, else 'LOW'."""
    num = cast_int(val, default=0)
    if num >= 100000:
        return "HIGH"
    elif num >= 60000:
        return "MID"
    return "LOW"


def compute_oracle_row(
    orig_row: dict[str, Any],
    mappings: list[ColumnMapping],
    lookup_data: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Compute expected target row values using the oracle transform library."""
    lookup_data = lookup_data or {}
    expected: dict[str, Any] = {}

    for m in mappings:
        src_col = m.source_column or m.new_column
        val = orig_row.get(src_col)

        trans = m.transform.lower() if m.transform else "passthrough"

        if trans in ("passthrough", "identity"):
            computed = val if val is not None else m.default_value
        elif trans in ("split_name_first", "first_name"):
            computed = split_name_first(val)
        elif trans in ("split_name_last", "last_name"):
            def_last = str(m.default_value) if m.default_value is not None else "N/A"
            computed = split_name_last(val, default=def_last)
        elif trans in ("strip_honorific", "clean_name"):
            computed = strip_honorific(val)
        elif trans in ("cast_int", "integer"):
            def_int = int(m.default_value) if m.default_value is not None else 0
            computed = cast_int(val, default=def_int)
        elif trans in ("cast_cents", "cents"):
            def_cents = int(m.default_value) if m.default_value is not None else 0
            computed = cast_cents(val, default=def_cents)
        elif trans in ("default_if_null", "coalesce"):
            computed = default_if_null(val, m.default_value)
        elif trans in ("map_via_lookup", "lookup"):
            tbl_lookups = lookup_data.get(m.lookup_table or "", {})
            if m.default_value is not None:
                def_lk = str(m.default_value)
            elif m.orphan_policy and m.orphan_policy.lower().startswith("default:"):
                def_lk = m.orphan_policy.split(":", 1)[1]
            elif m.orphan_policy and m.orphan_policy.lower() not in ("null", "none", "reject", "drop"):
                def_lk = str(m.orphan_policy)
            elif m.orphan_policy and m.orphan_policy.lower() in ("null", "none"):
                def_lk = None
            elif m.null_policy == "not_null":
                def_lk = "UNASSIGNED"
            else:
                def_lk = None
            computed = map_via_lookup(val, tbl_lookups, default=def_lk)
        elif trans in ("normalize_date", "date"):
            computed = normalize_date(val)
        elif trans in ("trim", "lower", "upper"):
            computed = trim_case(val, mode=trans)
        elif trans == "salary_band":
            num = cast_int(val, default=0)
            if num >= 100000:
                computed = "HIGH"
            elif num >= 60000:
                computed = "MID"
            else:
                computed = "LOW"
        else:
            # Custom transform without reference implementation
            computed = None

        if m.null_policy == "not_null" and computed is None and m.default_value is not None:
            computed = m.default_value

        expected[m.new_column] = computed

    return expected


def verify_rows_with_oracle(
    original_rows: list[dict[str, Any]],
    migrated_rows_by_pk: dict[str, dict[str, Any]],
    mappings: list[ColumnMapping],
    lookup_data: dict[str, dict[str, str]] | None = None,
    pk_column: str = "id",
) -> tuple[bool, dict[str, int], list[dict[str, Any]]]:
    """Verify all migrated rows against the oracle reference transforms."""
    mismatch_counts: dict[str, int] = {}
    failing_examples: list[dict[str, Any]] = []

    for orig in original_rows:
        pk_val = str(orig.get(pk_column, ""))
        actual = migrated_rows_by_pk.get(pk_val)
        if actual is None:
            continue

        expected = compute_oracle_row(orig, mappings, lookup_data)

        for col, exp_val in expected.items():
            if exp_val is None:
                # Custom transform not in oracle library
                continue
            act_val = actual.get(col)
            # Compare with type coercion flexibility (e.g. 1 == 1)
            if act_val != exp_val:
                mismatch_counts[col] = mismatch_counts.get(col, 0) + 1
                if len(failing_examples) < 3:
                    failing_examples.append({
                        "id": pk_val,
                        "column": col,
                        "expected": exp_val,
                        "actual": act_val,
                    })

    passed = len(mismatch_counts) == 0
    return passed, mismatch_counts, failing_examples
