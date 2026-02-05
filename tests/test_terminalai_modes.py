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


def test_choose_server_for_api_prefers_saved_server(monkeypatch):
    servers = [
        {"ip": "1.1.1.1", "nickname": "first", "apis": {"ollama": 11434}},
        {"ip": "2.2.2.2", "nickname": "second", "apis": {"ollama": 11434}},
    ]
    monkeypatch.setattr(TerminalAI, "load_servers", lambda api: servers)
    monkeypatch.setattr(
        TerminalAI,
        "MENU_STATE",
        {"ollama": {"ip": "2.2.2.2", "port": 11434}},
        raising=False,
    )

    selected = TerminalAI._choose_server_for_api("ollama", allow_back=True)

    assert selected == servers[1]


def test_choose_server_for_api_prompts_and_persists(monkeypatch):
    servers = [{"ip": "1.1.1.1", "nickname": "first", "apis": {"invokeai": 9090}}]
    remembered = []

    monkeypatch.setattr(TerminalAI, "load_servers", lambda api: servers)
    monkeypatch.setattr(TerminalAI, "MENU_STATE", {}, raising=False)
    monkeypatch.setattr(TerminalAI, "select_server", lambda options, allow_back=False: options[0])
    monkeypatch.setattr(
        TerminalAI,
        "_remember_server_selection",
        lambda api_type, server: remembered.append((api_type, server)),
    )

    selected = TerminalAI._choose_server_for_api("invokeai", allow_back=True)

    assert selected == servers[0]
    assert remembered == [("invokeai", servers[0])]


def test_run_chat_mode_back_from_model_selection_returns_to_main(monkeypatch):
    server = {"ip": "1.1.1.1", "nickname": "demo", "apis": {"ollama": 11434}}
    choose_calls = {"count": 0}

    def fake_choose_server(api_type, allow_back=True):
        choose_calls["count"] += 1
        return server

    monkeypatch.setattr(TerminalAI, "clear_screen", lambda *args, **kwargs: None)
    monkeypatch.setattr(TerminalAI, "_choose_server_for_api", fake_choose_server)
    monkeypatch.setattr(TerminalAI, "build_url", lambda *args, **kwargs: "http://demo")
    monkeypatch.setattr(TerminalAI, "fetch_models", lambda: ["model-a"])
    monkeypatch.setattr(TerminalAI, "select_model", lambda models: None)

    TerminalAI.run_chat_mode()

    assert choose_calls["count"] == 1


def test_run_image_mode_back_returns_to_main(monkeypatch):
    server = {"ip": "1.1.1.1", "nickname": "demo", "apis": {"invokeai": 9090}}
    choose_calls = {"count": 0}
    menu_calls = {"count": 0}

    class FakeClient:
        def __init__(self, ip, port, nickname, data_dir):
            self.ip = ip
            self.port = port
            self.nickname = nickname

        def check_health(self):
            return True

        def ensure_board(self, _name):
            return "board-id"

        def list_models(self):
            return [MagicMock(name="model")]

    def fake_choose_server(api_type, allow_back=True):
        choose_calls["count"] += 1
        return server

    def fake_menu(_header, options):
        menu_calls["count"] += 1
        return len(options) - 1

    monkeypatch.setattr(TerminalAI, "clear_screen", lambda *args, **kwargs: None)
    monkeypatch.setattr(TerminalAI, "_choose_server_for_api", fake_choose_server)
    monkeypatch.setattr(TerminalAI, "InvokeAIClient", FakeClient)
    monkeypatch.setattr(TerminalAI, "interactive_menu", fake_menu)

    TerminalAI.run_image_mode()

    assert choose_calls["count"] == 1
    assert menu_calls["count"] == 1
