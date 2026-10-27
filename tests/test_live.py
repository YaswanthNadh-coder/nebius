"""Live API tests against the real Nebius endpoint.

These tests are marked with `@pytest.mark.live` and skipped unless explicitly
requested and a valid non-placeholder NEBIUS_API_KEY is available.
"""

import pytest

from timemachine.config import get_api_key_info, is_placeholder_key


@pytest.mark.live
def test_live_nebius_models_resolution():
    """Verify models resolution against live Nebius endpoint."""
    key, source, masked = get_api_key_info()
    if is_placeholder_key(key):
        pytest.skip(f"Live test skipped: NEBIUS_API_KEY is not set or is a placeholder ({key}).")

    from timemachine.agent.llm import get_client, resolve_nebius_models

    client = get_client(key)
    fast_model, ultra_model, warnings = resolve_nebius_models(client)

    assert fast_model, "Expected fast_model to be resolved"
    assert ultra_model, "Expected ultra_model to be resolved"
