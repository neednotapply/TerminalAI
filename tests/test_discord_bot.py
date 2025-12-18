import json
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import discord_bot
from invoke_client import DEFAULT_SCHEDULER, InvokeAIModel


class FakeInvokeAIClient:
    def __init__(self, host, port, nickname, data_dir, image_path: Path):
        self.host = host
        self.port = port
        self.nickname = nickname
        self.data_dir = data_dir
        self.image_path = image_path
        self.submitted = None

    def ensure_board(self, board_name: str):
        return "board-1"

    def list_models(self):
        return [InvokeAIModel(name="Juggernaut XL v9", base="sdxl", key=None, raw={})]

    def generate_image(self, **kwargs):
        self.submitted = kwargs
        return {"path": self.image_path}


@pytest.mark.parametrize("width,height", [(512, 768), (9999, 16)])
def test_imagine_request_uses_default_scheduler(monkeypatch, tmp_path, width, height):
    image_file = tmp_path / "image.png"
    image_file.write_bytes(b"fake")
    client = FakeInvokeAIClient("1.1.1.1", 9090, "server", "data", image_file)

    def fake_client_ctor(*args, **kwargs):
        return client

    monkeypatch.setattr(discord_bot, "InvokeAIClient", fake_client_ctor)

    chat = discord_bot.ChatContext(
        endpoint={"ip": "1.1.1.1", "port": "9090", "id": "server"},
        mode="image-invokeai",
        model="Juggernaut XL v9",
    )

    _, image_path = discord_bot._send_imagine_request(
        chat,
        prompt="A beach at sunset",
        negative_prompt="",
        width=width,
        height=height,
        steps=30,
        cfg_scale=7.5,
    )

    assert client.submitted is not None
    assert image_path == image_file
    assert client.submitted["scheduler"] == DEFAULT_SCHEDULER
    assert client.submitted["width"] in {512, 2048}
    assert client.submitted["height"] in {768, 64}


def test_format_imagine_caption_removes_extra_whitespace():
    caption = discord_bot._format_imagine_caption("@User", "  A cat\n on a   mat  ")
    assert caption == "@User imagined A cat on a mat."


def _reset_discord_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discord_bot, "SESSIONS_PATH", tmp_path / "discord_sessions.json")
    discord_bot._persisted_state = {"contexts": {}, "sessions": {}}
    discord_bot._active_contexts = {}
    discord_bot._sessions = {}


class DummyChannel:
    def __init__(self) -> None:
        self.typing_entries = 0

    def typing(self):
        channel = self

        class _TypingContext:
            async def __aenter__(self):
                channel.typing_entries += 1

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return _TypingContext()


class DummyMessage:
    def __init__(self, content: str, author_id: int, mentions):
        self.content = content
        self.author = SimpleNamespace(id=author_id, bot=False)
        self.mentions = mentions
        self.channel = DummyChannel()
        self.replies = []

    async def reply(self, content: str, mention_author: bool = False):
        self.replies.append({"content": content, "mention_author": mention_author})


def test_context_persistence_across_restarts(monkeypatch, tmp_path):
    _reset_discord_state(tmp_path, monkeypatch)

    monkeypatch.setattr(
        discord_bot,
        "_load_endpoints",
        lambda mode: [{"ip": "1.1.1.1", "port": "8000", "id": "server"}],
    )

    chat = discord_bot.ChatContext(
        endpoint={"ip": "1.1.1.1", "port": "8000", "id": "server"},
        mode="llm-ollama",
        model="llama3",
        messages=[{"role": "user", "content": "Hello"}],
        available_models=["llama3"],
    )

    discord_bot._save_context(42, chat)
    discord_bot._active_contexts = {}

    restored = discord_bot._get_context(42, "llm-ollama")

    assert restored is not None
    assert restored.model == "llama3"
    assert restored.endpoint["ip"] == "1.1.1.1"
    assert restored.messages == chat.messages
    assert discord_bot.SESSIONS_PATH.exists()
    assert json.loads(discord_bot.SESSIONS_PATH.read_text())


def test_unavailable_context_is_not_restored(monkeypatch, tmp_path):
    _reset_discord_state(tmp_path, monkeypatch)

    monkeypatch.setattr(
        discord_bot,
        "_load_endpoints",
        lambda mode: [{"ip": "2.2.2.2", "port": "9000", "id": "other"}],
    )

    chat = discord_bot.ChatContext(
        endpoint={"ip": "1.1.1.1", "port": "8000", "id": "server"},
        mode="llm-ollama",
        model="llama3",
    )

    discord_bot._save_context(99, chat)
    discord_bot._active_contexts = {}

    restored = discord_bot._get_context(99, "llm-ollama")

    assert restored is None


def test_reset_session_clears_persisted_context(monkeypatch, tmp_path):
    _reset_discord_state(tmp_path, monkeypatch)

    monkeypatch.setattr(
        discord_bot,
        "_load_endpoints",
        lambda mode: [{"ip": "1.1.1.1", "port": "8000", "id": "server"}],
    )

    chat = discord_bot.ChatContext(
        endpoint={"ip": "1.1.1.1", "port": "8000", "id": "server"},
        mode="llm-ollama",
        model="llama3",
    )

    discord_bot._save_context(7, chat)
    assert str(7) in discord_bot._persisted_state["contexts"]

    discord_bot._reset_session(7)

    assert str(7) not in discord_bot._persisted_state.get("contexts", {})
    assert str(7) not in json.loads(discord_bot.SESSIONS_PATH.read_text()).get("contexts", {})


def test_strip_bot_mention_handles_multiple_formats():
    cleaned = discord_bot._strip_bot_mention("<@123> hi @TerminalAI", 123, "TerminalAI")
    assert cleaned == "hi"


def test_reply_to_mention_uses_existing_chat_context(monkeypatch, tmp_path):
    _reset_discord_state(tmp_path, monkeypatch)

    bot_user = SimpleNamespace(id=123, name="TerminalAI", display_name="TerminalAI")
    chat = discord_bot.ChatContext(
        endpoint={"ip": "1.1.1.1", "port": "8000", "id": "server"},
        mode="llm-ollama",
        model="llama3",
    )

    monkeypatch.setattr(
        discord_bot,
        "_get_context",
        lambda user_id, mode: chat,
    )

    message = DummyMessage("<@123> what is 2+2?", author_id=7, mentions=[bot_user])

    monkeypatch.setattr(
        discord_bot,
        "_send_chat_message",
        lambda chat_ctx, prompt: f"Echoed: {prompt}",
    )

    asyncio.run(discord_bot._reply_to_mention(message, bot_user))

    assert len(message.replies) == 1
    body = message.replies[0]["content"]
    assert "what is 2+2?" in body
    assert "Echoed: what is 2+2?" in body
    assert message.replies[0]["mention_author"] is False
