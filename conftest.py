"""Keep historical offline tests on their deterministic address configuration."""

import pytest
import config
from app.llm import UnhandledTurn


@pytest.fixture(autouse=True)
def address_extraction_test_mode(request, monkeypatch):
    # Enabled-path tests explicitly mock the transport in this module. Existing
    # parser tests continue to cover the supported opt-out mode without API calls.
    monkeypatch.setattr(
        config, "ADDRESS_LLM_EXTRACTION_ENABLED",
        request.node.path.name == "test_address_extraction.py",
    )
    # Historical parser tests cover the supported deterministic opt-out mode;
    # generic editor tests mock only the API transport and exercise default routing.
    monkeypatch.setattr(config, "BASKET_EDIT_LLM_ENABLED", request.node.path.name == "test_basket_edits.py")


@pytest.fixture(autouse=True)
def turn_model_follows_script(monkeypatch):
    """Offline tests keep the script unless they stub a specific model action."""
    monkeypatch.setattr(
        "app.dialog.resolve_unhandled_turn",
        lambda *_args, **_kwargs: UnhandledTurn(action="use_script"),
    )
