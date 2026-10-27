"""Configuration, environment loading, and settings for timemachine."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from timemachine.logging import get_logger

logger = get_logger("config")

PLACEHOLDER_KEY = "your_nebius_api_key_here"


class AgentError(Exception):
    """User-facing agent error with friendly explanation (no raw traceback)."""


def is_placeholder_key(key: str | None) -> bool:
    """Return True if *key* is empty or starts with 'your_'."""
    if not key or not isinstance(key, str):
        return True
    cleaned = key.strip().strip("'\"")
    return not cleaned or cleaned.startswith("your_")


def load_env_file(env_path: Path | str | None = None) -> str:
    """Safely load .env file with override=True and utf-8-sig encoding.

    Handles UTF-8 with BOM, UTF-16 fallback, and strips quotes/whitespace.
    Returns the loaded NEBIUS_API_KEY value.
    """
    if env_path is None:
        p = Path(__file__).resolve().parent.parent / ".env"
    elif isinstance(env_path, str):
        p = Path(env_path)
    else:
        p = env_path

    if not p.is_file():
        return os.environ.get("NEBIUS_API_KEY", "")

    # First attempt: load with utf-8-sig
    try:
        load_dotenv(dotenv_path=p, override=True, encoding="utf-8-sig")
    except Exception:
        # Fallback for UTF-16 or malformed encoding
        try:
            content = p.read_text(encoding="utf-16", errors="replace")
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip().lstrip("\ufeff")
                    v = v.strip().strip("'\"")
                    os.environ[k] = v
        except Exception as exc:
            logger.warning("Could not read .env file: %s", exc)
            return os.environ.get("NEBIUS_API_KEY", "")

    # Fix potential BOM key name in os.environ
    for k in list(os.environ.keys()):
        if k.startswith("\ufeff"):
            clean_k = k.lstrip("\ufeff")
            os.environ[clean_k] = os.environ.pop(k)

    # Clean quotes on NEBIUS_API_KEY if present
    if "NEBIUS_API_KEY" in os.environ:
        raw_val = os.environ["NEBIUS_API_KEY"].strip().strip("'\"")
        os.environ["NEBIUS_API_KEY"] = raw_val

    return os.environ.get("NEBIUS_API_KEY", "")


# Automatically load environment on import
load_env_file()


def get_api_key_info(override_key: str | None = None) -> tuple[str, str, str]:
    """Inspect and resolve API key, returning (clean_key, source, masked_repr).

    source is one of: 'sidebar', '.env', 'env var', or 'none'.
    """
    # 1. Check override passed explicitly (e.g. from UI sidebar)
    if override_key is not None and override_key.strip():
        cleaned = override_key.strip().strip("'\"")
        if not is_placeholder_key(cleaned):
            masked = f"{cleaned[:4]}... ({len(cleaned)} chars)"
            return cleaned, "sidebar", masked

    # 2. Re-read .env to check if file has key
    env_file_path = Path(__file__).resolve().parent.parent / ".env"
    file_key = ""
    if env_file_path.is_file():
        try:
            content = env_file_path.read_text(encoding="utf-8-sig", errors="replace")
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#") and line.startswith("NEBIUS_API_KEY="):
                    file_key = line.split("=", 1)[1].strip().strip("'\"")
                    break
        except Exception:
            pass

    if file_key and not is_placeholder_key(file_key):
        masked = f"{file_key[:4]}... ({len(file_key)} chars)"
        return file_key, ".env", masked

    # 3. Check environment variable
    env_var_key = os.environ.get("NEBIUS_API_KEY", "").strip().strip("'\"")
    if env_var_key and not is_placeholder_key(env_var_key):
        masked = f"{env_var_key[:4]}... ({len(env_var_key)} chars)"
        return env_var_key, "env var", masked

    return "", "none", "not set"


class Settings(BaseModel):
    """Typed application settings."""

    nebius_base_url: str = Field(
        default="https://api.tokenfactory.nebius.com/v1/",
        description="Nebius Token Factory endpoint URL",
    )
    default_fast_model: str = Field(
        default="nvidia/nemotron-3-super-120b-a12b",
        description="Fast drafting model (Nemotron Super)",
    )
    default_ultra_model: str = Field(
        default="nvidia/Nemotron-3-Ultra-550b-a55b",
        description="Deep repair & reasoning model (Nemotron Ultra)",
    )
    sandbox_timeout_s: float = Field(
        default=2.0,
        description="Sandbox query timeout deadline in seconds",
    )
    db_size_cap_mb: float = Field(
        default=200.0,
        description="Maximum source database file size allowed into RAM clone",
    )
    max_agent_steps: int = Field(
        default=8,
        description="Maximum agent tool-calling steps budget",
    )
    max_llm_tokens: int = Field(
        default=8192,
        description="Maximum completion tokens per LLM invocation",
    )
    llm_timeout_s: float = Field(
        default=90.0,
        description="HTTP client timeout for LLM requests",
    )
    agent_wall_clock_timeout_s: float = Field(
        default=180.0,
        description="Maximum total wall-clock budget in seconds for an agent execution run",
    )
    max_tool_rounds_per_step: int = Field(
        default=10,
        description="Maximum tool calling rounds per step",
    )



_SETTINGS_INSTANCE: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """Return singleton Settings instance."""
    global _SETTINGS_INSTANCE
    if _SETTINGS_INSTANCE is None or reload:
        _SETTINGS_INSTANCE = Settings()
    return _SETTINGS_INSTANCE
