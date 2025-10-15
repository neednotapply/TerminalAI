from unittest.mock import MagicMock, patch

import TerminalAI


def test_extract_mode_override_handles_new_aliases():
    cleaned, override, error = TerminalAI._extract_mode_override(
        ["prog", "--mode", "llm-ollama", "--verbose"]
    )

    assert cleaned == ["prog", "--verbose"]
    assert override == "llm-ollama"
    assert error is None


def test_extract_mode_override_supports_image_alias():
    cleaned, override, error = TerminalAI._extract_mode_override(["prog", "--mode=image"])

    assert cleaned == ["prog"]
    assert override == "image-invokeai"
    assert error is None


def test_extract_mode_override_supports_shodan():
    cleaned, override, error = TerminalAI._extract_mode_override(["prog", "--mode", "shodan"])

    assert cleaned == ["prog"]
    assert override == "shodan"
    assert error is None


def test_dispatch_mode_invokes_registered_handler():
    mock_handler = MagicMock()
    with patch.dict(TerminalAI.MODE_DISPATCH, {"llm-ollama": mock_handler}, clear=False):
        TerminalAI.dispatch_mode("llm-ollama")

    mock_handler.assert_called_once_with()


def test_dispatch_mode_returns_false_for_unknown_mode():
    with patch("builtins.print") as mock_print:
        result = TerminalAI.dispatch_mode("nonexistent-mode")

    assert result is False
    mock_print.assert_called()
