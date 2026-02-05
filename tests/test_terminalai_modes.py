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


def test_run_chat_mode_model_back_returns_to_main(monkeypatch):
    monkeypatch.setattr(TerminalAI, "clear_screen", lambda force=False: None)
    monkeypatch.setattr(
        TerminalAI,
        "_choose_server_for_api",
        lambda api_type, allow_back=True: {"ip": "1.1.1.1", "nickname": "srv", "apis": {"ollama": 11434}},
    )
    monkeypatch.setattr(TerminalAI, "build_url", lambda *_args, **_kwargs: "http://example")
    monkeypatch.setattr(TerminalAI, "fetch_models", lambda: ["model-a"])
    monkeypatch.setattr(TerminalAI, "select_model", lambda models: None)

    TerminalAI.run_chat_mode()


def test_run_image_mode_back_returns_to_main(monkeypatch):
    monkeypatch.setattr(TerminalAI, "clear_screen", lambda force=False: None)
    monkeypatch.setattr(
        TerminalAI,
        "_choose_server_for_api",
        lambda api_type, allow_back=True: {"ip": "1.1.1.1", "nickname": "srv", "apis": {"invokeai": 9090}},
    )

    class _Client:
        nickname = "srv"

        def __init__(self, *_args, **_kwargs):
            pass

        def check_health(self):
            return None

        def ensure_board(self, _name):
            return "terminalai"

        def list_models(self):
            return []

    monkeypatch.setattr(TerminalAI, "InvokeAIClient", _Client)
    monkeypatch.setattr(TerminalAI, "interactive_menu", lambda _header, _options: None)
    monkeypatch.setattr(TerminalAI, "get_input", lambda _prompt='': "")

    TerminalAI.run_image_mode()


def test_remember_server_selection_preserves_saved_model(monkeypatch):
    monkeypatch.setattr(
        TerminalAI,
        "MENU_STATE",
        {"ollama": {"ip": "1.1.1.1", "port": 11434, "model": "llama3"}},
        raising=False,
    )
    saved = []
    monkeypatch.setattr(TerminalAI, "_save_menu_state", lambda: saved.append(True))

    TerminalAI._remember_server_selection(
        "ollama", {"ip": "2.2.2.2", "apis": {"ollama": 11434}}
    )

    assert TerminalAI.MENU_STATE["ollama"]["ip"] == "2.2.2.2"
    assert TerminalAI.MENU_STATE["ollama"]["model"] == "llama3"
    assert saved


def test_preferred_ollama_model_returns_saved_match(monkeypatch):
    monkeypatch.setattr(
        TerminalAI,
        "MENU_STATE",
        {"ollama": {"ip": "1.1.1.1", "port": 11434, "model": "model-b"}},
        raising=False,
    )

    chosen = TerminalAI._get_preferred_ollama_model(["model-a", "model-b"])

    assert chosen == "model-b"


def test_preferred_invoke_model_uses_model_key(monkeypatch):
    models = [
        TerminalAI.InvokeAIModel(name="first", base="sd1", key="k1", raw={}),
        TerminalAI.InvokeAIModel(name="second", base="sdxl", key="k2", raw={}),
    ]
    monkeypatch.setattr(
        TerminalAI,
        "MENU_STATE",
        {"invokeai": {"ip": "1.1.1.1", "port": 9090, "model": "first", "model_key": "k2"}},
        raising=False,
    )

    chosen = TerminalAI._get_preferred_invoke_model(models)

    assert chosen is models[1]


def test_run_chat_mode_uses_saved_model_for_conversation_menu(monkeypatch):
    monkeypatch.setattr(TerminalAI, "clear_screen", lambda force=False: None)
    monkeypatch.setattr(
        TerminalAI,
        "_choose_server_for_api",
        lambda api_type, allow_back=True: {"ip": "1.1.1.1", "nickname": "srv", "apis": {"ollama": 11434}},
    )
    monkeypatch.setattr(TerminalAI, "build_url", lambda *_args, **_kwargs: "http://example")
    monkeypatch.setattr(TerminalAI, "fetch_models", lambda: ["saved-model"])
    monkeypatch.setattr(TerminalAI, "MENU_STATE", {"ollama": {"model": "saved-model"}}, raising=False)
    monkeypatch.setattr(TerminalAI, "has_conversations", lambda model: True)
    monkeypatch.setattr(
        TerminalAI,
        "select_conversation",
        lambda model: ("conv.json", [], [], None),
    )
    monkeypatch.setattr(TerminalAI, "chat_loop", lambda *_args, **_kwargs: "exit")

    def fail_select_model(_models):
        raise AssertionError("select_model should not be called when saved model is available")

    monkeypatch.setattr(TerminalAI, "select_model", fail_select_model)

    with patch("TerminalAI.sys.exit", side_effect=SystemExit):
        try:
            TerminalAI.run_chat_mode()
        except SystemExit:
            pass
