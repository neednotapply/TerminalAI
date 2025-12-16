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

import discord
from discord import app_commands
from discord.ext import commands

# Reuse the launcher menu configuration so Discord stays in sync.
from launcher import PROVIDER_HEADERS, PROVIDER_OPTIONS, TOP_LEVEL_OPTIONS

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


@dataclass
class MenuSession:
    """Per-user session that stores menu selections."""

    user_id: int
    history: List[str] = field(default_factory=list)
    top_key: Optional[str] = None

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
            if endpoints:
                view = discord.ui.View(timeout=300)
                view.add_item(EndpointSelect(label, endpoints, mode))
                await interaction.response.send_message(
                    f"Select a server for **{label}**:", view=view, ephemeral=True
                )
                await interaction.followup.send(self.session.history_message(), ephemeral=True)
                return

        if not interaction.response.is_done():
            defer_kwargs = {"ephemeral": True}
            if option.get("script") == "shodanscan.py":
                defer_kwargs["thinking"] = True
            await interaction.response.defer(**defer_kwargs)

        message = await _handle_option(option, interaction)
        await interaction.followup.send(message, ephemeral=True)
        await interaction.followup.send(self.session.history_message(), ephemeral=True)


class EndpointSelect(discord.ui.Select):
    def __init__(self, provider_label: str, endpoints: List[dict], mode: str):
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

    async def callback(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        idx = int(self.values[0])
        endpoint = self.endpoints[idx]

        name = endpoint.get("id") or "Selected endpoint"
        host = endpoint.get("ip") or endpoint.get("hostnames") or ""
        port = endpoint.get("port") or ""
        models = endpoint.get("available_models") or ""

        details = f"{name}: {host}:{port}" if host or port else name
        model_line = f"\nModels: {models}" if models else ""

        await interaction.response.send_message(
            f"{details}\nMode: `{self.mode}`{model_line}\n"
            "This action is interactive in the terminal. Run it locally with the selected server.",
            ephemeral=True,
        )


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
            await interaction.response.send_message(
                "Session closed. Use /terminalai to start again.", ephemeral=True
            )
            return

        if key not in PROVIDER_OPTIONS:
            await interaction.response.send_message(
                "That menu isn't available yet.", ephemeral=True
            )
            return

        view = discord.ui.View(timeout=300)
        view.add_item(ProviderSelect(key, self.session))
        await interaction.response.send_message(
            f"You picked **{key}**. Choose an action below:", view=view, ephemeral=True
        )


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
