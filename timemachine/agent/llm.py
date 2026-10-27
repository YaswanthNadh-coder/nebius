"""LLM client, model resolution, and safe execution wrappers."""

import os
import re

from openai import OpenAI

from timemachine.config import (
    AgentError,
    get_api_key_info,
    get_settings,
    is_placeholder_key,
)
from timemachine.logging import get_logger

logger = get_logger("agent.llm")

STATEMENT_KW_RE = re.compile(
    r"^\s*(CREATE|INSERT|DROP|ALTER|UPDATE|DELETE|WITH|PRAGMA|BEGIN|SELECT)\b",
    re.IGNORECASE,
)


def get_client(api_key_override: str | None = None) -> OpenAI:
    """Build an OpenAI-compatible client for Nebius Token Factory."""
    clean_key, source, masked = get_api_key_info(api_key_override)
    if is_placeholder_key(clean_key):
        raise AgentError(
            "No valid Nebius API key found. Paste your key in the sidebar or set it in .env."
        )

    settings = get_settings()
    return OpenAI(
        base_url=settings.nebius_base_url,
        api_key=clean_key,
        timeout=settings.llm_timeout_s,
        max_retries=2,
    )


def resolve_nebius_models(client: OpenAI) -> tuple[str, str, list[str]]:
    """Resolve available Nemotron Super and Ultra models from Nebius Token Factory.

    Returns: (fast_model_id, ultra_model_id, warnings_list).
    """
    settings = get_settings()
    fast_env = os.environ.get("NEBIUS_FAST_MODEL")
    ultra_env = os.environ.get("NEBIUS_ULTRA_MODEL")

    if fast_env and ultra_env:
        return fast_env, ultra_env, []

    try:
        models_data = client.models.list().data
        available_ids = [m.id for m in models_data]
    except Exception as exc:
        raise AgentError(f"Failed to query models from Nebius API: {exc}") from exc

    # Filter out embedding, rerank, reward, guard, safety models
    _EXCLUDE = re.compile(r"embed|rerank|reward|guard|safety", re.IGNORECASE)
    chat_ids = [mid for mid in available_ids if not _EXCLUDE.search(mid)]

    warnings: list[str] = []

    # 1. Resolve fast model (Nemotron Super)
    fast_model = fast_env
    if not fast_model:
        if settings.default_fast_model in chat_ids:
            fast_model = settings.default_fast_model
        else:
            super_candidates = [m for m in chat_ids if re.search(r"nemotron.*super", m, re.I)]
            if super_candidates:
                fast_model = super_candidates[0]
            else:
                nano_candidates = [m for m in chat_ids if re.search(r"nemotron.*nano", m, re.I)]
                if nano_candidates:
                    fast_model = nano_candidates[0]
                    warnings.append(
                        f"Nemotron Super not available on account; falling back to Nano ({fast_model})."
                    )

    # 2. Resolve ultra model (Nemotron Ultra)
    ultra_model = ultra_env
    if not ultra_model:
        if settings.default_ultra_model in chat_ids:
            ultra_model = settings.default_ultra_model
        else:
            ultra_candidates = [m for m in chat_ids if re.search(r"nemotron.*ultra", m, re.I)]
            if ultra_candidates:
                ultra_model = ultra_candidates[0]

    if not fast_model and not ultra_model:
        raise AgentError(
            f"No suitable Nemotron chat models found on Nebius account. "
            f"Available model IDs: {available_ids}"
        )

    if not fast_model and ultra_model:
        fast_model = ultra_model
    if not ultra_model and fast_model:
        ultra_model = fast_model
        warnings.append(
            f"Nemotron Ultra not available; repair will use {fast_model}."
        )

    return fast_model, ultra_model, warnings


def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
) -> tuple[str, list[dict] | None]:
    """Execute LLM call with complete error translation and finish_reason handling.

    Returns: (text_content, tool_calls_list).
    """
    settings = get_settings()
    tokens = max_tokens or settings.max_llm_tokens

    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": tokens,
    }
    if tools:
        kwargs["tools"] = tools

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        name = type(exc).__name__
        msg = str(exc)
        if "401" in msg or "unauthorized" in msg.lower():
            raise AgentError(
                "Authentication failed (HTTP 401). Check that your NEBIUS_API_KEY is active and valid."
            ) from exc
        if "429" in msg or "rate" in msg.lower():
            raise AgentError(
                "Rate limit reached (HTTP 429). Please wait a few seconds and retry."
            ) from exc
        if "timeout" in name.lower() or "timeout" in msg.lower():
            raise AgentError(
                "Nebius API request timed out (90s). The model service may be experiencing high load."
            ) from exc
        if "connect" in name.lower() or "connect" in msg.lower():
            raise AgentError(
                "Unable to connect to Nebius Token Factory. Please check your network connection."
            ) from exc
        raise AgentError(f"API Error ({name}): {msg}") from exc

    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise AgentError(
            "Model output was truncated (max_tokens limit reached). Try breaking down the migration goal."
        )

    # Safely extract message content and tool calls
    msg_obj = choice.message
    content = msg_obj.content or ""

    tool_calls = None
    if getattr(msg_obj, "tool_calls", None):
        tool_calls = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg_obj.tool_calls
        ]

    return content, tool_calls


def clean_sql(raw_text: str | None) -> str:
    """Extract executable SQL statements from LLM output.

    - Handles <think>...</think> and unclosed <think> blocks.
    - Concatenates multiple code blocks in order.
    - Handles fences tagged with sql, sqlite (case-insensitive) or untagged.
    - Strips unterminated fence openings.
    - Drops leading conversational prose prior to statement keywords.
    """
    if not raw_text:
        return ""

    text = raw_text.replace("\r\n", "\n")

    # Strip closed and unclosed <think> reasoning blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.strip()
    if not text:
        return ""

    # Line-anchored fence regex matching sql, sqlite, or untagged
    fence_re = re.compile(
        r"^```(?:sql|sqlite)?\s*$\n(.*?)^```\s*$",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    blocks = fence_re.findall(text)
    if blocks:
        valid_blocks = [b.strip() for b in blocks if b.strip()]
        if valid_blocks:
            return "\n\n".join(valid_blocks)

    # Handle unterminated fence opening
    open_fence = re.compile(r"^```(?:sql|sqlite)?\s*$", re.MULTILINE | re.IGNORECASE)
    if open_fence.search(text):
        text = open_fence.sub("", text).strip()

    # Drop leading conversational prose lines
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if STATEMENT_KW_RE.match(line):
            return "\n".join(lines[i:]).strip()

    return ""
