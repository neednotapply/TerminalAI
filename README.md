# BorrowedCompute

BorrowedCompute is a retro-styled client for chatting with Ollama models, generating images from community InvokeAI servers, and for discovering new public instances via Shodan. A launcher provides a unified entry point on Linux, macOS, and Windows.

## Installation

1. **Clone the repository**

   ```bash
   git clone <REPO_URL>
   cd BorrowedCompute
   ```

2. **Install Python dependencies**

   ```bash
   cd scripts
   python -m pip install -r requirements.txt
   ```

   On Windows, use the Python launcher if needed:

   ```bat
   py -m pip install -r requirements.txt
   ```

3. **Prepare data files**
   - Copy `data/ollama.endpoints.example.csv` to `data/ollama.endpoints.csv` and add any known Ollama servers.
   - Copy `data/invoke.endpoints.example.csv` to `data/invoke.endpoints.csv` for InvokeAI hosts.
   - (Optional) Create `data/config.json` to store your Shodan API key and Discord bot settings (`DISCORD_BOT_TOKEN`, `BORROWEDCOMPUTE_WORKING_DIR`). A template is available at `data/config.example.json`.

## Usage

### Launcher

Use the launcher for a graphical menu with matrix rain effects.

- **Linux/macOS**: `./launcher.sh`
- **Windows**: `launcher.bat`

The launcher now groups actions by API. Pick **LLM Chat** or **Image Generation**, then choose a provider-specific task such as launching BorrowedCompute or running a Shodan scan scoped to that API. Endpoint refreshes still update the corresponding CSV files under `data/`.

The **Configure** menu is organized by capability: **Configure Chat** contains Ollama server/model selection, while **Configure Image Generation** contains InvokeAI and Automatic1111 server/model selection.

Press the highlighted number to choose an option. The launcher restores the console on exit.

### Chat Client

Run the chat client directly if desired:

```bash
python3 scripts/BorrowedCompute.py
```

The client loads servers from `data/ollama.endpoints.csv` (for chat) and `data/invoke.endpoints.csv` (for image generation), pings them to sort by latency, and stores conversations under `data/conversations`. Logs produced with the `/print` command are written to `data/logs`.

### Image Generation

InvokeAI and Automatic1111 image generation are available from the main BorrowedCompute menu. The client discovers API-compatible servers, lists the models each host exposes, and walks through prompt configuration (including width/height, steps, CFG scale, scheduler or sampler, and optional seed). Generated images are downloaded to `data/images/` along with per-render metadata.

BorrowedCompute renders image previews directly in the terminal using [chafa.py](https://github.com/GuardKenzie/chafa.py) (with [Pillow](https://python-pillow.org/)) and will fall back to the [chafa](https://hpjansson.org/chafa/) CLI if the Python bindings are unavailable.

### Discord Bot (optional)

An optional Discord bot mirrors the launcher menus with per-user, ephemeral interactions. Both chat and image-generation flows are available in Discord through the same menu groupings used by the launcher.

1. Install dependencies if you haven't already:

   ```bash
   cd scripts
   pip3 install -r requirements.txt
   ```

2. Add your Discord settings to `data/config.json`:

   ```json
   {
     "DISCORD_BOT_TOKEN": "<your token>",
     "BORROWEDCOMPUTE_WORKING_DIR": "optional/path/if/scripts/aren't/in/PWD"
   }
   ```

3. Start the bot:

   ```bash
   python3 discord_bot.py
   ```

The bot exposes a `/config` command with a compact, private configuration menu. Choose **Shodan**, **Chat**, or **Imagine**, then select from servers and models that pass live compatibility checks. Send prompts with `/chat` (Ollama) or `/imagine` (InvokeAI or Automatic1111). Shodan scans run directly from Discord.

Server and model selections are shared through `data/menu_state.json`. Configure a server and model in the terminal app, exit it, and the Discord bot will use those selections after validating availability. Discord selections are written back to the same file; `data/discord_preferences.json` remains a compatibility fallback for older saved preferences.

### Shodan Scan

Scan for new supported servers and verify existing ones:

```bash
python3 scripts/shodanscan.py [--debug] [--limit N] [--existing-limit N]
```

The script requires a Shodan API key. Results are appended to `data/ollama.endpoints.csv` and `data/invoke.endpoints.csv`, enriched with metadata such as hostnames, organisation, ISP, location, and (for InvokeAI) available models. Existing entries are checked for availability and ping time.

## Development

Run a basic syntax check on all Python scripts:

```bash
python3 -m py_compile scripts/*.py
```

## Data Layout

- `data/ollama.endpoints.csv` – list of known Ollama endpoints.
- `data/invoke.endpoints.csv` – list of known InvokeAI endpoints.
- `data/images/` – InvokeAI image outputs and metadata files.
- `data/conversations/` – per-model conversation history.
- `data/logs/` – saved transcripts.
- `data/config.json` – optional configuration (`SHODAN_API_KEY`).
- `data/discord_preferences.json` – persisted Discord server and model choices.

## License

This project is provided as-is without any warranty. Use it responsibly.
