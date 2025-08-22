# TerminalAI

TerminalAI is a retro-styled client for chatting with Ollama models and for discovering new public instances via Shodan. A launcher provides a unified entry point on Linux, macOS, and Windows.

## Installation

1. **Clone the repository**

   ```bash
   git clone <REPO_URL>
   cd TerminalAI
   ```

2. **Install Python dependencies**

   ```bash
   cd data
   pip3 install -r requirements.txt
   ```

3. **Prepare data files**
   - Copy `data/endpoints.example.csv` to `data/endpoints.csv` and add any known servers.
   - (Optional) Create `data/config.json` or set the `SHODAN_API_KEY` environment variable for Shodan queries. A template is available at `data/config.example.json`.

## Usage

### Launcher

Use the launcher for a graphical menu with matrix rain effects.

- **Linux/macOS**: `./launcher.sh`
- **Windows**: `launcher.bat`

The launcher offers two actions:

1. **Start TerminalAI** – chat with a selected Ollama server.
2. **Scan Shodan** – update `data/endpoints.csv` with public instances.

Press `1` or `2` to choose an action. The launcher restores the console on exit.

### Chat Client

Run the chat client directly if desired:

```bash
python3 scripts/TerminalAI.py
```

The client loads servers from `data/endpoints.csv`, pings them to sort by latency, and stores conversations under `data/conversations`. Logs produced with the `/print` command are written to `data/logs`.

### Shodan Scan

Scan for new Ollama servers and verify existing ones:

```bash
python3 scripts/shodanscan.py [--verbose] [--limit N] [--existing-limit N]
```

The script requires a Shodan API key. Results are appended to `data/endpoints.csv` and enriched with metadata such as hostnames, organisation, ISP, and location. Existing entries are checked for availability and ping time.

## Development

Run a basic syntax check on all Python scripts:

```bash
python3 -m py_compile scripts/*.py
```

## Data Layout

- `data/endpoints.csv` – list of known endpoints.
- `data/conversations/` – per-model conversation history.
- `data/logs/` – saved transcripts.
- `data/config.json` – optional configuration (`SHODAN_API_KEY`).

## License

This project is provided as-is without any warranty. Use it responsibly.
