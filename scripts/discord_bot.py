"""Discord bot wrapper for navigating TerminalAI menus.

This bot mirrors the launcher menu structure and exposes it through
Discord application commands. Responses are sent ephemerally so each
user receives a private menu and history of the options they select.

Environment variables:
- DISCORD_BOT_TOKEN: required Discord bot token.

Permissions:
- Enable the **Message Content Intent** for the bot in the Discord developer portal.

The bot is intentionally light-weight and only runs actions that are
non-interactive inside Discord (for example Shodan scans). Interactive
flows such as the TUI chat client are presented with the command that
would normally be run locally.
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

import discord
from discord import app_commands
from discord.ext import commands

# Reuse the launcher menu configuration so Discord stays in sync.
from launcher import PROVIDER_HEADERS, PROVIDER_OPTIONS, TOP_LEVEL_OPTIONS
from invoke_client import InvokeAIClient, InvokeAIClientError
from TerminalAI import (
    TERMINALAI_BOARD_ID_ERROR_MESSAGE,
    TERMINALAI_BOARD_NAME,
    _normalize_board_id,
    _is_valid_terminalai_board_id,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = DATA_DIR / "config.json"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


CONFIG = _load_config()


def _working_directory() -> Path:
    return Path(__file__).resolve().parent


def _parse_models(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    normalized = raw.replace("|", ";").replace(",", ";").replace("\n", ";")
    return [model.strip() for model in normalized.split(";") if model.strip()]


def _first_value(raw: Optional[str]) -> str:
    if not raw:
        return ""
    for separator in (";", ",", " "):
        if separator in raw:
            return raw.split(separator, 1)[0].strip()
    return str(raw).strip()


def _compose_content(header: str, session: MenuSession, chat: Optional["ChatContext"] = None) -> str:
    parts = [header]
    if chat:
        parts.append(chat.describe())
    parts.append(session.history_message())
    return "\n\n".join(part for part in parts if part)


@dataclass
class MenuSession:
    """Per-user session that stores menu selections."""

    user_id: int
    history: List[str] = field(default_factory=list)
    top_key: Optional[str] = None
    mode: Optional[str] = None
    endpoints: List[dict] = field(default_factory=list)
    provider_label: Optional[str] = None
    chat: Optional["ChatContext"] = None

    def log(self, entry: str) -> None:
        self.history.append(entry)

    def history_message(self) -> str:
        if not self.history:
            return "You haven't picked anything yet."
        bullet_points = "\n".join(f"• {item}" for item in self.history[-10:])
        return f"Your recent picks (most recent last):\n{bullet_points}"


_sessions: Dict[int, MenuSession] = {}


def _reset_session(user_id: int) -> MenuSession:
    """Create a fresh session for a user, clearing any prior history."""

    session = MenuSession(user_id=user_id)
    _sessions[user_id] = session
    return session


@dataclass
class ChatContext:
    endpoint: dict
    mode: str
    model: Optional[str] = None
    messages: List[dict] = field(default_factory=list)
    available_models: List[str] = field(default_factory=list)
    board_notice: Optional[str] = None
    board_error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.available_models:
            self.available_models = _parse_models(self.endpoint.get("available_models"))
        if not self.model and self.available_models:
            self.model = self.available_models[0]

    @property
    def base_url(self) -> Optional[str]:
        host = _first_value(self.endpoint.get("ip") or self.endpoint.get("hostnames"))
        port = _first_value(self.endpoint.get("port"))
        if not host or not port:
            return None
        return f"http://{host}:{port}"

    def describe(self) -> str:
        name = self.endpoint.get("id") or "Selected endpoint"
        host = _first_value(self.endpoint.get("ip") or self.endpoint.get("hostnames"))
        port = _first_value(self.endpoint.get("port"))
        details = f"{name}: {host}:{port}" if host or port else name
        model_line = self.model or "Select a model to begin chatting."
        transcript = self._transcript_preview()
        parts = [
            f"Server: {details}",
            f"Mode: `{self.mode}`",
            f"Active model: {model_line}",
            transcript,
        ]
        return "\n".join(part for part in parts if part)

    def _transcript_preview(self) -> str:
        if not self.messages:
            return ""
        preview_lines = []
        for entry in self.messages[-6:]:
            role = entry.get("role")
            content = (entry.get("content") or "").replace("`", "'")
            label = "You" if role == "user" else "Assistant"
            truncated = (content[:180] + "…") if len(content) > 180 else content
            preview_lines.append(f"{label}: {truncated}")
        return "\n".join(preview_lines)


async def _update_menu_message(
    interaction: discord.Interaction, content: str, view: Optional[discord.ui.View]
) -> None:
    if interaction.response.is_done():
        await interaction.edit_original_response(content=content, view=view)
    else:
        await interaction.response.edit_message(content=content, view=view)


async def _run_shodan_scan(option: dict, interaction: discord.Interaction) -> str:
    """Execute a Shodan scan action and return a trimmed result string."""

    script_path = _working_directory() / option["script"]
    cmd = [sys.executable, str(script_path)] + option.get("extra_args", [])
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(_working_directory()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
    except asyncio.TimeoutError:
        process.kill()
        return "Scan timed out before completing."

    output = stdout.decode().strip()
    errors = stderr.decode().strip()
    combined = output
    if errors:
        combined = f"{combined}\n\nErrors:\n{errors}" if combined else f"Errors:\n{errors}"
    if not combined:
        return "Scan finished without output."
    return combined[-1900:]


async def _handle_option(option: dict, interaction: discord.Interaction) -> str:
    """Dispatch a provider option to an action string for Discord."""

    script = option.get("script")
    if not script:
        return "Selected option has no linked action."

    if script == "shodanscan.py":
        return await _run_shodan_scan(option, interaction)

    script_path = _working_directory() / script
    arg_list = " ".join(option.get("extra_args", []))
    command = f"{sys.executable} {script_path} {arg_list}".strip()
    return (
        "This action is interactive in the terminal.\n"
        "Run it locally with:\n"
        f"`{command}`"
    )


def _mode_from_args(extra_args: List[str]) -> Optional[str]:
    """Extract the value passed to a --mode flag from an argument list."""

    for idx, arg in enumerate(extra_args):
        if arg == "--mode" and idx + 1 < len(extra_args):
            return extra_args[idx + 1]
        if arg.startswith("--mode="):
            return arg.split("=", 1)[1]
    return None


def _load_endpoints(mode: str) -> List[dict]:
    """Load endpoints CSV rows for a given mode, falling back to examples."""

    api_mapping = {
        "llm-ollama": DATA_DIR / "ollama.endpoints.csv",
        "image-invokeai": DATA_DIR / "invoke.endpoints.csv",
        "image-automatic1111": DATA_DIR / "automatic1111.endpoints.csv",
    }
    example_mapping = {
        "llm-ollama": DATA_DIR / "ollama.endpoints.example.csv",
        "image-invokeai": DATA_DIR / "invoke.endpoints.example.csv",
        "image-automatic1111": DATA_DIR / "automatic1111.endpoints.example.csv",
    }

    path = api_mapping.get(mode)
    fallback = example_mapping.get(mode)
    csv_path = path if path and path.exists() else fallback
    if not csv_path or not csv_path.exists():
        return []

    rows: List[dict] = []
    try:
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
    except OSError:
        return []
    return rows


def _fetch_endpoint_models(chat: ChatContext) -> List[str]:
    chat.board_error = None
    chat.board_notice = None
    if chat.mode == "image-invokeai":
        return _fetch_invoke_models(chat)

    base_url = chat.base_url
    if not base_url:
        return []
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return []

    models: List[str] = []
    for entry in payload.get("models") or payload.get("data") or []:
        if isinstance(entry, dict):
            candidate = entry.get("id") or entry.get("name") or entry.get("model")
        else:
            candidate = str(entry)
        if candidate:
            models.append(str(candidate))
    return [model for model in models if model]


def _extract_chat_content(payload: dict) -> Optional[str]:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message") or first.get("delta") or {}
            if isinstance(message, dict):
                content = message.get("content")
                if content:
                    return content
    for key in ("message", "response", "content", "text"):
        candidate = payload.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _fetch_invoke_models(chat: ChatContext) -> List[str]:
    host = _first_value(chat.endpoint.get("ip") or chat.endpoint.get("hostnames"))
    port = _first_value(chat.endpoint.get("port"))
    if not host or not port:
        chat.board_error = "Selected server is missing connection details."
        return []

    try:
        client = InvokeAIClient(host, int(port), chat.endpoint.get("id") or host, DATA_DIR)
    except (TypeError, ValueError) as exc:
        chat.board_error = f"Unable to connect to InvokeAI server: {exc}"
        return []

    try:
        board_id = client.ensure_board(TERMINALAI_BOARD_NAME)
    except InvokeAIClientError as exc:
        chat.board_error = str(exc)
        return []
    except requests.RequestException as exc:
        chat.board_error = f"Network error while verifying InvokeAI board: {exc}"
        return []

    if not _is_valid_terminalai_board_id(board_id):
        chat.board_error = TERMINALAI_BOARD_ID_ERROR_MESSAGE
        return []

    normalized_id = _normalize_board_id(board_id) or board_id
    chat.board_notice = (
        f"Images will be saved to board {TERMINALAI_BOARD_NAME} (id: {normalized_id})."
    )

    try:
        models = client.list_models()
    except InvokeAIClientError as exc:
        chat.board_error = str(exc)
        return []
    except requests.RequestException as exc:
        chat.board_error = f"Network error while fetching InvokeAI models: {exc}"
        return []

    names = [model.name for model in models if getattr(model, "name", None)]
    return [name for name in names if name]


def _send_chat_message(chat: ChatContext, prompt: str) -> str:
    base_url = chat.base_url
    if not base_url:
        return "The selected server is missing connection details."

    chat_paths = ["/v1/chat/completions", "/api/chat", "/chat"]
    payload = {
        "model": chat.model,
        "messages": chat.messages + [{"role": "user", "content": prompt}],
        "stream": False,
    }

    last_error = ""
    for path in chat_paths:
        try:
            resp = requests.post(f"{base_url}{path}", json=payload, timeout=30)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            content = _extract_chat_content(data)
            if not content:
                last_error = "Server returned an empty response."
                continue
            chat.messages.append({"role": "user", "content": prompt})
            chat.messages.append({"role": "assistant", "content": content})
            return content
        except requests.RequestException as exc:  # pragma: no cover - network failures
            last_error = str(exc)
        except ValueError:
            last_error = "Invalid JSON response"
    return f"Failed to chat with the server. {last_error or 'No supported endpoints responded.'}"


def _trim_discord_message(content: str, limit: int = 1800) -> str:
    return content if len(content) <= limit else content[: limit - 1] + "…"


class ProviderSelect(discord.ui.Select):
    def __init__(self, top_key: str, session: MenuSession):
        options_config = PROVIDER_OPTIONS.get(top_key, [])
        choices = [
            discord.SelectOption(label=opt.get("label", "Unnamed"), value=str(idx))
            for idx, opt in enumerate(options_config)
        ]
        placeholder = PROVIDER_HEADERS.get(top_key, "Pick an action")
        super().__init__(
            placeholder=placeholder,
            min_values=1,
            max_values=1,
            options=choices,
        )
        self.top_key = top_key
        self.session = session

    async def callback(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        idx = int(self.values[0])
        provider_options = PROVIDER_OPTIONS.get(self.top_key, [])
        option = provider_options[idx]
        label = option.get("label", "Unknown option")
        self.session.log(f"{self.top_key} -> {label}")

        if option.get("script") == "TerminalAI.py":
            mode = _mode_from_args(option.get("extra_args", [])) or ""
            endpoints = _load_endpoints(mode)
            self.session.mode = mode
            self.session.endpoints = endpoints
            self.session.provider_label = label
            self.session.chat = None
            if endpoints:
                view = discord.ui.View(timeout=300)
                view.add_item(EndpointSelect(label, endpoints, mode, self.session))
                content = _compose_content(
                    f"Select a server for **{label}**:", self.session
                )
                await _update_menu_message(interaction, content, view)
                return
            content = _compose_content(
                f"No servers found for **{label}**.", self.session
            )
            await _update_menu_message(interaction, content, None)
            return

        if not interaction.response.is_done():
            defer_kwargs = {"ephemeral": True}
            if option.get("script") == "shodanscan.py":
                defer_kwargs["thinking"] = True
            await interaction.response.defer(**defer_kwargs)

        message = await _handle_option(option, interaction)
        followup_view = discord.ui.View(timeout=300)
        followup_view.add_item(ProviderSelect(self.top_key, self.session))
        content = _compose_content(_trim_discord_message(message), self.session)
        await interaction.edit_original_response(content=content, view=followup_view)


class EndpointSelect(discord.ui.Select):
    def __init__(
        self, provider_label: str, endpoints: List[dict], mode: str, session: MenuSession
    ):
        options = []
        for idx, endpoint in enumerate(endpoints):
            label = endpoint.get("id") or endpoint.get("hostnames") or "Unnamed"
            description = endpoint.get("org") or endpoint.get("isp") or ""
            options.append(
                discord.SelectOption(
                    label=label[:100],
                    value=str(idx),
                    description=description[:100] or None,
                )
            )
            if len(options) >= 25:
                break

        placeholder = f"Select a {provider_label} server"
        super().__init__(placeholder=placeholder, min_values=1, max_values=1, options=options)
        self.endpoints = endpoints
        self.mode = mode
        self.provider_label = provider_label
        self.session = session

    async def callback(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        idx = int(self.values[0])
        endpoint = self.endpoints[idx]

        chat = ChatContext(endpoint=endpoint, mode=self.mode)
        self.session.chat = chat
        self.session.log(f"{self.provider_label} -> {endpoint.get('id') or 'endpoint'}")
        self.session.endpoints = self.endpoints

        await interaction.response.defer(ephemeral=True, thinking=True)

        fetched_models = await asyncio.to_thread(_fetch_endpoint_models, chat)
        if chat.board_error:
            content = _compose_content(chat.board_error, self.session, chat)
            await _update_menu_message(interaction, content, ChatView(self.session))
            return

        if fetched_models:
            chat.available_models = fetched_models
            if chat.model not in fetched_models:
                chat.model = fetched_models[0]
            model_message = "Server selected. Pick a model and start chatting."
        elif chat.available_models:
            model_message = "Server selected. Using configured models for this server."
        else:
            model_message = (
                "Server selected, but no models are available yet. "
                "Use **Refresh models** after the server finishes loading."
            )

        if chat.board_notice:
            model_message = f"{chat.board_notice}\n{model_message}"

        content = _compose_content(
            model_message, self.session, chat
        )
        await _update_menu_message(interaction, content, ChatView(self.session))


class ModelSelect(discord.ui.Select):
    def __init__(self, session: MenuSession):
        self.session = session
        chat = session.chat
        options = []
        if chat:
            models = chat.available_models
            more_count = max(0, len(models) - 25)
            for model in models[:25]:
                options.append(
                    discord.SelectOption(
                        label=model[:100],
                        value=model,
                        default=bool(chat.model and chat.model == model),
                    )
                )
            placeholder = (
                "Pick a model (showing first 25)"
                if more_count
                else "Pick a model"
            )
        else:
            placeholder = "Pick a model"
        if not options:
            options.append(
                discord.SelectOption(
                    label="No models available", value="no-models", default=True
                )
            )

        super().__init__(
            placeholder=placeholder,
            options=options,
            min_values=1,
            max_values=1,
            disabled=not chat or not chat.available_models,
        )

    async def callback(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        chat = self.session.chat
        if not chat:
            content = _compose_content(
                "Select a server before choosing a model.", self.session, None
            )
            await _update_menu_message(interaction, content, ChatView(self.session))
            return

        new_model = self.values[0]
        if not new_model:
            content = _compose_content(
                "No models are available for this server yet.", self.session, chat
            )
            await _update_menu_message(interaction, content, ChatView(self.session))
            return

        chat.model = new_model
        self.session.log(f"Model -> {new_model}")
        content = _compose_content(
            "Model selected. Send a prompt to chat.", self.session, chat
        )
        await _update_menu_message(interaction, content, ChatView(self.session))


class RefreshModelsButton(discord.ui.Button):
    def __init__(self, session: MenuSession):
        super().__init__(label="Refresh models", style=discord.ButtonStyle.secondary)
        self.session = session

    async def callback(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        chat = self.session.chat
        if not chat:
            content = _compose_content(
                "Select a server to refresh its models.", self.session, None
            )
            await _update_menu_message(interaction, content, ChatView(self.session))
            return

        await interaction.response.defer(ephemeral=True, thinking=True)
        models = await asyncio.to_thread(_fetch_endpoint_models, chat)
        if chat.board_error:
            content = _compose_content(chat.board_error, self.session, chat)
            await interaction.edit_original_response(
                content=_trim_discord_message(content), view=ChatView(self.session)
            )
            return

        if models:
            chat.available_models = models
            if chat.model not in models:
                chat.model = models[0]
            message = "Model list refreshed from the server."
            self.session.log("Models refreshed")
        else:
            message = "Failed to refresh models from the server."

        if chat.board_notice:
            message = f"{chat.board_notice}\n{message}"

        content = _compose_content(message, self.session, chat)
        await interaction.edit_original_response(
            content=_trim_discord_message(content), view=ChatView(self.session)
        )


class ChangeServerButton(discord.ui.Button):
    def __init__(self, session: MenuSession):
        super().__init__(label="Change server", style=discord.ButtonStyle.secondary)
        self.session = session

    async def callback(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        if not self.session.endpoints:
            content = _compose_content(
                "No server list available. Pick a menu again with /terminalai.",
                self.session,
                self.session.chat,
            )
            await _update_menu_message(interaction, content, ChatView(self.session))
            return

        mode = self.session.mode or (self.session.chat.mode if self.session.chat else "")
        view = discord.ui.View(timeout=300)
        view.add_item(
            EndpointSelect(
                self.session.provider_label or "Server",
                self.session.endpoints,
                mode,
                self.session,
            )
        )
        content = _compose_content(
            f"Select a server for **{self.session.provider_label or 'server'}**:",
            self.session,
            self.session.chat,
        )
        await _update_menu_message(interaction, content, view)


class PromptModal(discord.ui.Modal):
    def __init__(self, session: MenuSession):
        super().__init__(title="Send a prompt")
        self.session = session
        self.prompt: discord.ui.TextInput = discord.ui.TextInput(
            label="Prompt",
            style=discord.TextStyle.long,
            required=True,
            max_length=800,
            placeholder="Ask a question for the selected model",
        )
        self.add_item(self.prompt)

    async def on_submit(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        chat = self.session.chat
        if not chat or not chat.model:
            content = _compose_content(
                "Select a server and model before chatting.", self.session, chat
            )
            await _update_menu_message(interaction, content, ChatView(self.session))
            return

        prompt_text = str(self.prompt.value).strip()
        if not prompt_text:
            content = _compose_content("Prompt cannot be empty.", self.session, chat)
            await _update_menu_message(interaction, content, ChatView(self.session))
            return

        await interaction.response.defer(ephemeral=True, thinking=True)
        self.session.log(f"Prompt -> {chat.model}")
        reply = await asyncio.to_thread(_send_chat_message, chat, prompt_text)
        header = f"You: {prompt_text}\nAssistant: {reply}"
        content = _compose_content(_trim_discord_message(header), self.session, chat)
        await interaction.edit_original_response(content=content, view=ChatView(self.session))


class PromptButton(discord.ui.Button):
    def __init__(self, session: MenuSession):
        super().__init__(label="Send prompt", style=discord.ButtonStyle.primary)
        self.session = session

    async def callback(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        chat = self.session.chat
        if not chat:
            content = _compose_content(
                "Select a server before sending prompts.", self.session, None
            )
            await _update_menu_message(interaction, content, ChatView(self.session))
            return
        if not chat.model:
            content = _compose_content(
                "Choose a model before chatting.", self.session, chat
            )
            await _update_menu_message(interaction, content, ChatView(self.session))
            return

        await interaction.response.send_modal(PromptModal(self.session))


class ChatView(discord.ui.View):
    def __init__(self, session: MenuSession):
        super().__init__(timeout=300)
        self.session = session

        if session.chat:
            self.add_item(ModelSelect(session))
        self.add_item(RefreshModelsButton(session))
        self.add_item(PromptButton(session))
        self.add_item(ChangeServerButton(session))


class TopMenuView(discord.ui.View):
    def __init__(self, session: MenuSession):
        super().__init__(timeout=300)
        self.session = session

        choices = [
            discord.SelectOption(label=item.get("label", ""), value=item.get("key", ""))
            for item in TOP_LEVEL_OPTIONS
            if item.get("key")
        ]
        self.menu = discord.ui.Select(
            placeholder="Pick a menu",
            options=choices,
            min_values=1,
            max_values=1,
        )
        self.menu.callback = self.on_select  # type: ignore[assignment]
        self.add_item(self.menu)

    async def on_select(self, interaction: discord.Interaction) -> None:
        key = self.menu.values[0]
        self.session.top_key = key
        self.session.log(f"Selected top-level menu: {key}")

        if key == "exit":
            await _update_menu_message(
                interaction, "Session closed. Use /terminalai to start again.", None
            )
            return

        if key not in PROVIDER_OPTIONS:
            await _update_menu_message(
                interaction, _compose_content("That menu isn't available yet.", self.session), self
            )
            return

        view = discord.ui.View(timeout=300)
        view.add_item(ProviderSelect(key, self.session))
        content = _compose_content(
            f"You picked **{key}**. Choose an action below:", self.session
        )
        await _update_menu_message(interaction, content, view)


class TerminalAIDiscord(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self) -> None:
        self.tree.add_command(start_menu)
        await self.tree.sync()


bot = TerminalAIDiscord()


@app_commands.command(name="terminalai", description="Open the TerminalAI menu in Discord")
async def start_menu(interaction: discord.Interaction) -> None:
    session = _reset_session(interaction.user.id)
    view = TopMenuView(session)
    await interaction.response.send_message(
        "Pick a menu to get started.", view=view, ephemeral=True
    )


def main() -> None:
    token = CONFIG.get("DISCORD_BOT_TOKEN") or os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "DISCORD_BOT_TOKEN is required to run the Discord bot. "
            "Set it in data/config.json or as an environment variable."
        )
    bot.run(token)


if __name__ == "__main__":
    main()
