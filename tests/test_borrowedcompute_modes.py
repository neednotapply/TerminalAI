from unittest.mock import MagicMock, patch
from ollama_compat import OllamaProbeResult
import json

import BorrowedCompute


def test_extract_mode_override_handles_new_aliases():
    cleaned, override, error = BorrowedCompute._extract_mode_override(
        ["prog", "--mode", "llm-ollama", "--verbose"]
    )

    assert cleaned == ["prog", "--verbose"]
    assert override == "llm-ollama"
    assert error is None


def test_extract_mode_override_supports_image_alias():
    cleaned, override, error = BorrowedCompute._extract_mode_override(["prog", "--mode=image"])

    assert cleaned == ["prog"]
    assert override == "image"
    assert error is None


def test_extract_mode_override_supports_shodan():
    cleaned, override, error = BorrowedCompute._extract_mode_override(["prog", "--mode", "shodan"])

    assert cleaned == ["prog"]
    assert override == "shodan"
    assert error is None


def test_dispatch_mode_invokes_registered_handler():
    mock_handler = MagicMock()
    with patch.dict(BorrowedCompute.MODE_DISPATCH, {"llm-ollama": mock_handler}, clear=False):
        BorrowedCompute.dispatch_mode("llm-ollama")

    mock_handler.assert_called_once_with()


def test_dispatch_mode_returns_false_for_unknown_mode():
    with patch("builtins.print") as mock_print:
        result = BorrowedCompute.dispatch_mode("nonexistent-mode")

    assert result is False
    mock_print.assert_called()


def test_run_shodan_scan_injects_api_type(monkeypatch):
    commands = []

    monkeypatch.setattr(BorrowedCompute, "clear_screen", lambda force=False: None)
    monkeypatch.setattr(BorrowedCompute, "DEBUG_MODE", True, raising=False)
    monkeypatch.setattr(BorrowedCompute.sys, "argv", ["BorrowedCompute.py", "--foo"])

    def fake_call(cmd):
        commands.append(cmd)
        return 0

    monkeypatch.setattr(BorrowedCompute.subprocess, "call", fake_call)

    BorrowedCompute.run_shodan_scan("invokeai")

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
    monkeypatch.setattr(BorrowedCompute, "load_servers", lambda api: servers)
    monkeypatch.setattr(
        BorrowedCompute,
        "MENU_STATE",
        {"ollama": {"ip": "2.2.2.2", "port": 11434}},
        raising=False,
    )

    selected = BorrowedCompute._choose_server_for_api("ollama", allow_back=True)

    assert selected == servers[1]


def test_choose_server_for_api_prompts_and_persists(monkeypatch):
    servers = [{"ip": "1.1.1.1", "nickname": "first", "apis": {"invokeai": 9090}}]
    remembered = []

    monkeypatch.setattr(BorrowedCompute, "load_servers", lambda api: servers)
    monkeypatch.setattr(BorrowedCompute, "MENU_STATE", {}, raising=False)
    monkeypatch.setattr(BorrowedCompute, "select_server", lambda options, allow_back=False: options[0])
    monkeypatch.setattr(
        BorrowedCompute,
        "_remember_server_selection",
        lambda api_type, server: remembered.append((api_type, server)),
    )

    selected = BorrowedCompute._choose_server_for_api("invokeai", allow_back=True)

    assert selected == servers[0]
    assert remembered == [("invokeai", servers[0])]


def test_run_chat_mode_model_back_returns_to_main(monkeypatch):
    monkeypatch.setattr(BorrowedCompute, "clear_screen", lambda force=False: None)
    monkeypatch.setattr(
        BorrowedCompute,
        "_choose_server_for_api",
        lambda api_type, allow_back=True: {"ip": "1.1.1.1", "nickname": "srv", "apis": {"ollama": 11434}},
    )
    monkeypatch.setattr(BorrowedCompute, "build_url", lambda *_args, **_kwargs: "http://example")
    monkeypatch.setattr(BorrowedCompute, "fetch_models", lambda: ["model-a"])
    monkeypatch.setattr(BorrowedCompute, "select_model", lambda models: None)

    BorrowedCompute.run_chat_mode()


def test_run_image_mode_back_returns_to_main(monkeypatch):
    monkeypatch.setattr(BorrowedCompute, "clear_screen", lambda force=False: None)
    monkeypatch.setattr(
        BorrowedCompute,
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
            return "borrowedcompute"

        def list_models(self):
            return []

    monkeypatch.setattr(BorrowedCompute, "InvokeAIClient", _Client)
    monkeypatch.setattr(BorrowedCompute, "interactive_menu", lambda _header, _options: None)
    monkeypatch.setattr(BorrowedCompute, "get_input", lambda _prompt='': "")

    BorrowedCompute.run_image_mode()


def test_remember_server_selection_preserves_saved_model(monkeypatch):
    monkeypatch.setattr(
        BorrowedCompute,
        "MENU_STATE",
        {"ollama": {"ip": "1.1.1.1", "port": 11434, "model": "llama3"}},
        raising=False,
    )
    saved = []
    monkeypatch.setattr(BorrowedCompute, "_save_menu_state", lambda: saved.append(True))

    BorrowedCompute._remember_server_selection(
        "ollama", {"ip": "2.2.2.2", "apis": {"ollama": 11434}}
    )

    assert BorrowedCompute.MENU_STATE["ollama"]["ip"] == "2.2.2.2"
    assert BorrowedCompute.MENU_STATE["ollama"]["model"] == "llama3"
    assert saved


def test_preferred_ollama_model_returns_saved_match(monkeypatch):
    monkeypatch.setattr(
        BorrowedCompute,
        "MENU_STATE",
        {"ollama": {"ip": "1.1.1.1", "port": 11434, "model": "model-b"}},
        raising=False,
    )

    chosen = BorrowedCompute._get_preferred_ollama_model(["model-a", "model-b"])

    assert chosen == "model-b"


def test_preferred_invoke_model_uses_model_key(monkeypatch):
    models = [
        BorrowedCompute.InvokeAIModel(name="first", base="sd1", key="k1", raw={}),
        BorrowedCompute.InvokeAIModel(name="second", base="sdxl", key="k2", raw={}),
    ]
    monkeypatch.setattr(
        BorrowedCompute,
        "MENU_STATE",
        {"invokeai": {"ip": "1.1.1.1", "port": 9090, "model": "first", "model_key": "k2"}},
        raising=False,
    )

    chosen = BorrowedCompute._get_preferred_invoke_model(models)

    assert chosen is models[1]


def test_run_chat_mode_uses_saved_model_for_conversation_menu(monkeypatch):
    monkeypatch.setattr(BorrowedCompute, "clear_screen", lambda force=False: None)
    monkeypatch.setattr(
        BorrowedCompute,
        "_choose_server_for_api",
        lambda api_type, allow_back=True: {"ip": "1.1.1.1", "nickname": "srv", "apis": {"ollama": 11434}},
    )
    monkeypatch.setattr(BorrowedCompute, "build_url", lambda *_args, **_kwargs: "http://example")
    monkeypatch.setattr(BorrowedCompute, "fetch_models", lambda: ["saved-model"])
    monkeypatch.setattr(BorrowedCompute, "MENU_STATE", {"ollama": {"model": "saved-model"}}, raising=False)
    monkeypatch.setattr(BorrowedCompute, "has_conversations", lambda model: True)
    monkeypatch.setattr(
        BorrowedCompute,
        "select_conversation",
        lambda model: ("conv.json", [], [], None),
    )
    monkeypatch.setattr(BorrowedCompute, "chat_loop", lambda *_args, **_kwargs: "exit")

    def fail_select_model(_models):
        raise AssertionError("select_model should not be called when saved model is available")

    monkeypatch.setattr(BorrowedCompute, "select_model", fail_select_model)

    with patch("BorrowedCompute.sys.exit", side_effect=SystemExit):
        try:
            BorrowedCompute.run_chat_mode()
        except SystemExit:
            pass


def test_run_chat_mode_conversation_back_returns_to_main_menu(monkeypatch):
    monkeypatch.setattr(BorrowedCompute, "clear_screen", lambda force=False: None)
    monkeypatch.setattr(
        BorrowedCompute,
        "_choose_server_for_api",
        lambda api_type, allow_back=True: {"ip": "1.1.1.1", "nickname": "srv", "apis": {"ollama": 11434}},
    )
    monkeypatch.setattr(BorrowedCompute, "build_url", lambda *_args, **_kwargs: "http://example")
    monkeypatch.setattr(BorrowedCompute, "fetch_models", lambda: ["saved-model"])
    monkeypatch.setattr(BorrowedCompute, "MENU_STATE", {"ollama": {"model": "saved-model"}}, raising=False)

    has_conv_calls = iter([True, False])
    monkeypatch.setattr(BorrowedCompute, "has_conversations", lambda model: next(has_conv_calls))
    monkeypatch.setattr(BorrowedCompute, "select_conversation", lambda model: ("back", None, None, None))
    monkeypatch.setattr(BorrowedCompute, "chat_loop", lambda *_args, **_kwargs: "exit")

    select_model_calls = []

    def fake_select_model(models):
        select_model_calls.append(models)
        return None

    monkeypatch.setattr(BorrowedCompute, "select_model", fake_select_model)

    forgotten = []

    def fake_forget_model_selection(api_type):
        forgotten.append(api_type)
        BorrowedCompute.MENU_STATE.get(api_type, {}).pop("model", None)

    monkeypatch.setattr(BorrowedCompute, "_forget_model_selection", fake_forget_model_selection)

    with patch("BorrowedCompute.sys.exit", side_effect=SystemExit):
        BorrowedCompute.run_chat_mode()

    assert forgotten == ["ollama"]
    assert select_model_calls == []


def test_fetch_models_falls_back_to_ollama_tags(monkeypatch):
    monkeypatch.setattr(BorrowedCompute, "SERVER_URL", "http://example", raising=False)
    monkeypatch.setattr(
        BorrowedCompute,
        "probe_ollama_endpoint",
        lambda endpoint: OllamaProbeResult(
            "accessible", models=["llama3", "phi3"], chat_models=["llama3", "phi3"]
        ),
    )
    monkeypatch.setattr(BorrowedCompute, "_ensure_model_entry", lambda model: ({}, False))
    monkeypatch.setattr(BorrowedCompute, "update_model_capabilities_from_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(BorrowedCompute, "_save_model_capabilities", lambda: None)

    assert BorrowedCompute.fetch_models() == ["llama3", "phi3"]


def test_fetch_models_returns_empty_when_all_endpoints_fail(monkeypatch):
    monkeypatch.setattr(BorrowedCompute, "SERVER_URL", "http://example", raising=False)
    monkeypatch.setattr(
        BorrowedCompute,
        "probe_ollama_endpoint",
        lambda endpoint: OllamaProbeResult("unavailable", reason="down"),
    )

    assert BorrowedCompute.fetch_models() == []


def test_shared_selection_round_trip(monkeypatch, tmp_path):
    state_path = tmp_path / "menu_state.json"
    monkeypatch.setattr(BorrowedCompute, "MENU_STATE_FILE", state_path)
    monkeypatch.setattr(BorrowedCompute, "MENU_STATE", {})

    BorrowedCompute.save_shared_selection(
        "ollama",
        {"ip": "1.2.3.4", "port": "11434"},
        model="llama3:latest",
    )

    assert json.loads(state_path.read_text()) == {
        "ollama": {"ip": "1.2.3.4", "port": 11434, "model": "llama3:latest"}
    }
    assert BorrowedCompute.get_shared_selection("ollama") == {
        "ip": "1.2.3.4",
        "port": 11434,
        "model": "llama3:latest",
    }


def test_configure_ollama_model_uses_preselected_server(monkeypatch):
    server = {"ip": "1.1.1.1", "nickname": "srv", "apis": {"ollama": 11434}}
    monkeypatch.setattr(
        BorrowedCompute,
        "_get_configured_server_for_api",
        lambda api_type, label: server,
    )
    monkeypatch.setattr(BorrowedCompute, "build_url", lambda *_args, **_kwargs: "http://example")
    monkeypatch.setattr(BorrowedCompute, "fetch_models", lambda: ["model-a"])
    monkeypatch.setattr(BorrowedCompute, "select_model", lambda models: "model-a")

    remembered = []
    monkeypatch.setattr(
        BorrowedCompute,
        "_remember_model_selection",
        lambda api_type, model_name, model_key=None: remembered.append((api_type, model_name, model_key)),
    )
    monkeypatch.setattr(BorrowedCompute, "get_input", lambda _prompt='': "")
    monkeypatch.setattr(
        BorrowedCompute,
        "_configure_api_server",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("server prompt should not run from model menu")),
    )

    BorrowedCompute._configure_ollama_model()

    assert remembered == [("ollama", "model-a", None)]


def test_configure_invokeai_model_requires_preselected_server(monkeypatch):
    monkeypatch.setattr(
        BorrowedCompute,
        "_get_configured_server_for_api",
        lambda api_type, label: None,
    )
    monkeypatch.setattr(
        BorrowedCompute,
        "InvokeAIClient",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("client should not be created without configured server")),
    )

    BorrowedCompute._configure_invokeai_model()


def test_build_url_reuses_scanner_connection_metadata():
    server = {
        "ip": "1.2.3.4",
        "apis": {"ollama": 443},
        "connections": {
            "ollama": {"scheme": "https", "host": "ollama.example.com"}
        },
    }

    assert BorrowedCompute.build_url(server, "ollama") == (
        "https://ollama.example.com:443"
    )


def test_extract_chat_text_supports_native_ollama_response():
    payload = {
        "model": "llama3",
        "message": {"role": "assistant", "content": "Hello from Ollama"},
        "done": True,
    }

    assert BorrowedCompute.extract_chat_text(payload) == "Hello from Ollama"


def test_preferred_automatic1111_model_uses_saved_title(monkeypatch):
    models = [
        BorrowedCompute.Automatic1111Model("one", "Model One", None, {}),
        BorrowedCompute.Automatic1111Model("two", "Model Two", None, {}),
    ]
    monkeypatch.setattr(
        BorrowedCompute,
        "MENU_STATE",
        {"automatic1111": {"model": "Model Two"}},
    )

    assert BorrowedCompute._get_preferred_automatic1111_model(models) is models[1]
