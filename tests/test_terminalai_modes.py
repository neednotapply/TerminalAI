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


def test_run_shodan_scan_injects_api_type(monkeypatch):
    commands = []

    monkeypatch.setattr(TerminalAI, "clear_screen", lambda force=False: None)
    monkeypatch.setattr(TerminalAI, "DEBUG_MODE", True, raising=False)
    monkeypatch.setattr(TerminalAI.sys, "argv", ["TerminalAI.py", "--foo"])

    def fake_call(cmd):
        commands.append(cmd)
        return 0

    monkeypatch.setattr(TerminalAI.subprocess, "call", fake_call)

    TerminalAI.run_shodan_scan("invokeai")

    assert commands
    cmd = commands[0]
    assert "--api-type" in cmd
    api_index = cmd.index("--api-type")
    assert api_index + 1 < len(cmd)
    assert cmd[api_index + 1] == "invokeai"
