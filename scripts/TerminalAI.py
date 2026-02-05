import requests
import time
import datetime
import os
import sys
import shutil
import pandas as pd
import select
import json
import re
import threading
import socket
import subprocess
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple
from pathlib import Path
from rain import rain
from invoke_client import (
    DEFAULT_CFG_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_SCHEDULER,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    FLUX_DEFAULT_SCHEDULER,
    UNCATEGORIZED_BOARD_ID,
    InvokeAIClient,
    InvokeAIClientError,
    InvokeAIModel,
)

from automatic1111_client import (
    Automatic1111Client,
    Automatic1111ClientError,
    Automatic1111Model,
)

try:
    import chafa
except ImportError:  # pragma: no cover - optional dependency
    chafa = None

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover - optional dependency
    PILImage = None

if os.name == "nt":
    import msvcrt
else:
    import curses
    import curses.textpad
    import termios
    import tty


def _read_escape_sequence() -> str:
    """Read the remainder of an ANSI escape sequence."""

    if os.name == "nt":
        seq = ""
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            seq += ch
            if ch.isalpha() or ch in "~":
                break
        return seq

    fd = sys.stdin.fileno()
    seq = ""
    deadline = time.monotonic() + 0.1
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        ready, _, _ = select.select([sys.stdin], [], [], remaining)
        if not ready:
            continue
        ch = os.read(fd, 1).decode(errors="ignore")
        if not ch:
            break
        seq += ch
        if ch.isalpha() or ch in "~":
            break
    return seq

# ANSI colors
GREEN = "\033[38;2;5;249;0m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
WARNING = "\033[38;2;204;176;0m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
AI_COLOR = "\033[32m"

TERMINALAI_BOARD_NAME = "TerminalAI"
TERMINALAI_BOARD_RESOLUTION_ERROR = (
    f"Failed to resolve InvokeAI board '{TERMINALAI_BOARD_NAME}'"
)
TERMINALAI_BOARD_ID_ERROR_MESSAGE = (
    f"InvokeAI server did not provide a valid id for board {TERMINALAI_BOARD_NAME}."
)

BATCH_STATUS_TIMEOUT = 120.0
BATCH_STATUS_POLL_INTERVAL = 2.0


AUTOMATIC1111_DEFAULT_WIDTH = 512
AUTOMATIC1111_DEFAULT_HEIGHT = 512
AUTOMATIC1111_DEFAULT_STEPS = 30
AUTOMATIC1111_DEFAULT_CFG_SCALE = 7.0
AUTOMATIC1111_DEFAULT_SAMPLER = "Euler a"


def _normalize_board_id(board_id: Any) -> Optional[str]:
    if not isinstance(board_id, str):
        return None
    candidate = board_id.strip()
    return candidate or None


def _is_valid_terminalai_board_id(board_id: Any) -> bool:
    normalized = _normalize_board_id(board_id)
    return normalized is not None and normalized != UNCATEGORIZED_BOARD_ID

# Seconds to keep the "Hack The Planet" splash visible before continuing.
CONNECTING_SCREEN_DURATION = 0.5

DEBUG_FLAGS = {"--debug", "--verbose"}
DEBUG_MODE = any(flag in sys.argv for flag in DEBUG_FLAGS)

MODE_ALIASES = {
    "chat": "chat",
    "imagine": "imagine",
    "configure": "configure",
    "ollama": "llm-ollama",
    "llm": "llm",
    "llm-ollama": "llm-ollama",
    "image": "image-invokeai",
    "img": "image-invokeai",
    "invoke": "image-invokeai",
    "invokeai": "image-invokeai",
    "image-invokeai": "image-invokeai",
    "image-automatic1111": "image-automatic1111",
    "automatic1111": "image-automatic1111",
    "a1111": "image-automatic1111",
    "shodan": "shodan",
    "scan": "shodan",
}
MODE_LABELS = {
    "chat": "Chat",
    "imagine": "Imagine",
    "configure": "Configure",
    "llm": "LLM Chat",
    "llm-ollama": "LLM Chat (Ollama)",
    "image": "Image Generation",
    "image-invokeai": "Image Generation (InvokeAI)",
    "image-automatic1111": "Image Generation (Automatic1111)",
    "shodan": "Shodan Scan",
}
MODE_OVERRIDE = None
MODE_OVERRIDE_ERROR = None

MODE_DISPATCH: Dict[str, Callable[[], None]] = {}


def _extract_mode_override(argv: List[str]) -> Tuple[List[str], Optional[str], Optional[str]]:
    """Return cleaned argv, mode override, and error message for a given argv."""

    if not argv:
        return [], None, None

    program = argv[0]
    args = list(argv[1:])
    cleaned = [program]
    i = 0
    override = None
    error = None
    while i < len(args):
        arg = args[i]
        value = None
        if arg == "--mode":
            if i + 1 < len(args):
                value = args[i + 1]
                i += 2
            else:
                error = "--mode flag requires a value"
                i += 1
                continue
        elif arg.startswith("--mode="):
            value = arg.split("=", 1)[1]
            i += 1
        else:
            cleaned.append(arg)
            i += 1
            continue

        key = (value or "").strip().lower()
        if not key:
            error = "--mode flag requires a value"
            override = None
        elif key in MODE_ALIASES:
            override = MODE_ALIASES[key]
            error = None
        else:
            error = f"Unknown mode '{value}'"
            override = None

    return cleaned, override, error


def _parse_mode_override():
    """Consume --mode arguments from sys.argv and configure overrides."""

    global MODE_OVERRIDE, MODE_OVERRIDE_ERROR

    cleaned, override, error = _extract_mode_override(sys.argv)
    if cleaned:
        sys.argv[:] = cleaned
    MODE_OVERRIDE = override
    MODE_OVERRIDE_ERROR = error


_parse_mode_override()

def clear_screen(force=False):
    if not DEBUG_MODE or force:
        os.system("cls" if os.name == "nt" else "clear")

def heat_color(ping):
    if ping is None or ping == float("inf"):
        return RED
    max_ping = 1000.0
    p = max(0.0, min(ping, max_ping)) / max_ping
    r = int(255 * p)
    g = int(255 * (1 - p))
    return f"\033[38;2;{r};{g};0m"

SERVER_URL = ""
selected_server = None
selected_api = None
IDLE_TIMEOUT = 30
REQUEST_TIMEOUT = 60
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LEGACY_CSV_PATH = DATA_DIR / "endpoints.csv"
OLLAMA_CSV_PATH = DATA_DIR / "ollama.endpoints.csv"
INVOKE_CSV_PATH = DATA_DIR / "invoke.endpoints.csv"
AUTOMATIC1111_CSV_PATH = DATA_DIR / "automatic1111.endpoints.csv"
CSV_PATHS = {
    "ollama": OLLAMA_CSV_PATH,
    "invokeai": INVOKE_CSV_PATH,
    "automatic1111": AUTOMATIC1111_CSV_PATH,
}
BASE_COLUMNS = [
    "id",
    "ip",
    "port",
    "scan_date",
    "verified",
    "verification_date",
    "is_active",
    "inactive_reason",
    "last_check_date",
    "api_type",
    "hostnames",
    "org",
    "isp",
    "city",
    "region",
    "country",
    "latitude",
    "longitude",
    "ping",
]
COLUMN_ORDER = {
    "ollama": BASE_COLUMNS,
    "invokeai": BASE_COLUMNS + ["available_models"],
    "automatic1111": BASE_COLUMNS + ["available_models"],
}
CONV_DIR = DATA_DIR / "conversations"
LOG_DIR = DATA_DIR / "logs"

MODEL_CAPABILITIES_FILE = DATA_DIR / "model_capabilities.json"
MENU_STATE_FILE = DATA_DIR / "menu_state.json"
MODEL_CAPABILITIES: Dict[str, Dict[str, Any]] = {}
MENU_STATE: Dict[str, Dict[str, Any]] = {}
EMBED_MODEL_KEYWORDS = ("embed", "embedding")


def _utcnow_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_model_entry(name: str) -> Tuple[Dict[str, Any], bool]:
    info = MODEL_CAPABILITIES.setdefault(name, {})
    changed = False
    if "first_seen" not in info:
        info["first_seen"] = _utcnow_iso()
        changed = True
    return info, changed


def _load_model_capabilities() -> None:
    global MODEL_CAPABILITIES
    try:
        with MODEL_CAPABILITIES_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return
    except (OSError, ValueError) as exc:
        if DEBUG_MODE:
            print(f"{YELLOW}Failed to load model capabilities: {exc}{RESET}")
        return

    if not isinstance(data, dict):
        return

    cleaned: Dict[str, Dict[str, Any]] = {}
    for name, info in data.items():
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(info, dict):
            continue
        entry: Dict[str, Any] = {}
        for key in ("is_embedding", "confirmed", "inferred"):
            value = info.get(key)
            if isinstance(value, bool):
                entry[key] = value
        for key in ("first_seen", "updated_at"):
            value = info.get(key)
            if isinstance(value, str):
                entry[key] = value
        cleaned[name] = entry

    MODEL_CAPABILITIES.clear()
    MODEL_CAPABILITIES.update(cleaned)


def _save_model_capabilities() -> None:
    try:
        MODEL_CAPABILITIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Dict[str, Any]] = {}
        for name, info in MODEL_CAPABILITIES.items():
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(info, dict):
                continue
            entry: Dict[str, Any] = {}
            for key in ("is_embedding", "confirmed", "inferred"):
                value = info.get(key)
                if isinstance(value, bool):
                    entry[key] = value
            for key in ("first_seen", "updated_at"):
                value = info.get(key)
                if isinstance(value, str):
                    entry[key] = value
            payload[name] = entry
        with MODEL_CAPABILITIES_FILE.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    except OSError as exc:
        if DEBUG_MODE:
            print(f"{YELLOW}Failed to save model capabilities: {exc}{RESET}")


_load_model_capabilities()


def _load_menu_state() -> None:
    global MENU_STATE

    MENU_STATE = {}
    try:
        with MENU_STATE_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return
    except (OSError, ValueError):
        return

    if not isinstance(data, dict):
        return

    for api_type in ("ollama", "invokeai"):
        entry = data.get(api_type)
        if not isinstance(entry, dict):
            continue
        ip = entry.get("ip")
        port = entry.get("port")
        if not isinstance(ip, str) or not ip.strip():
            continue
        try:
            numeric_port = int(port)
        except (TypeError, ValueError):
            continue
        cleaned_entry: Dict[str, Any] = {"ip": ip.strip(), "port": numeric_port}
        model = entry.get("model")
        if isinstance(model, str) and model.strip():
            cleaned_entry["model"] = model.strip()
        model_key = entry.get("model_key")
        if isinstance(model_key, str) and model_key.strip():
            cleaned_entry["model_key"] = model_key.strip()
        MENU_STATE[api_type] = cleaned_entry


def _save_menu_state() -> None:
    payload: Dict[str, Dict[str, Any]] = {}
    for api_type, entry in MENU_STATE.items():
        if not isinstance(entry, dict):
            continue
        ip = entry.get("ip")
        port = entry.get("port")
        if not isinstance(ip, str) or not ip.strip():
            continue
        try:
            numeric_port = int(port)
        except (TypeError, ValueError):
            continue
        cleaned_entry: Dict[str, Any] = {"ip": ip.strip(), "port": numeric_port}
        model = entry.get("model")
        if isinstance(model, str) and model.strip():
            cleaned_entry["model"] = model.strip()
        model_key = entry.get("model_key")
        if isinstance(model_key, str) and model_key.strip():
            cleaned_entry["model_key"] = model_key.strip()
        payload[api_type] = cleaned_entry

    try:
        MENU_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with MENU_STATE_FILE.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    except OSError:
        return


def _remember_server_selection(api_type: str, server: Dict[str, Any]) -> None:
    port = server.get("apis", {}).get(api_type)
    if port is None:
        return
    try:
        numeric_port = int(port)
    except (TypeError, ValueError):
        return
    existing = MENU_STATE.get(api_type)
    entry: Dict[str, Any] = {}
    if isinstance(existing, dict):
        entry.update(existing)
    entry["ip"] = server.get("ip", "")
    entry["port"] = numeric_port
    MENU_STATE[api_type] = entry
    _save_menu_state()


def _remember_model_selection(api_type: str, model_name: str, model_key: Optional[str] = None) -> None:
    if not isinstance(model_name, str) or not model_name.strip():
        return
    existing = MENU_STATE.get(api_type)
    entry: Dict[str, Any] = {}
    if isinstance(existing, dict):
        entry.update(existing)
    entry["model"] = model_name.strip()
    if isinstance(model_key, str) and model_key.strip():
        entry["model_key"] = model_key.strip()
    else:
        entry.pop("model_key", None)
    MENU_STATE[api_type] = entry
    _save_menu_state()


def _forget_model_selection(api_type: str) -> None:
    entry = MENU_STATE.get(api_type)
    if not isinstance(entry, dict):
        return
    changed = False
    if "model" in entry:
        entry.pop("model", None)
        changed = True
    if "model_key" in entry:
        entry.pop("model_key", None)
        changed = True
    if changed:
        MENU_STATE[api_type] = entry
        _save_menu_state()


def _forget_server_selection(api_type: str) -> None:
    if api_type in MENU_STATE:
        MENU_STATE.pop(api_type, None)
        _save_menu_state()


def _get_preferred_server(api_type: str, servers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    selection = MENU_STATE.get(api_type)
    if not isinstance(selection, dict):
        return None

    selected_ip = selection.get("ip")
    selected_port = selection.get("port")
    if not isinstance(selected_ip, str):
        return None
    try:
        selected_port_value = int(selected_port)
    except (TypeError, ValueError):
        return None

    for server in servers:
        if server.get("ip") != selected_ip:
            continue
        port = server.get("apis", {}).get(api_type)
        try:
            server_port = int(port)
        except (TypeError, ValueError):
            continue
        if server_port == selected_port_value:
            return server
    return None


def _get_preferred_ollama_model(models: List[str]) -> Optional[str]:
    entry = MENU_STATE.get("ollama")
    if not isinstance(entry, dict):
        return None
    preferred = entry.get("model")
    if not isinstance(preferred, str):
        return None
    preferred_name = preferred.strip()
    if not preferred_name:
        return None
    for model in models:
        if isinstance(model, str) and model == preferred_name:
            return model
    return None


def _get_preferred_invoke_model(models: List[InvokeAIModel]) -> Optional[InvokeAIModel]:
    entry = MENU_STATE.get("invokeai")
    if not isinstance(entry, dict):
        return None

    preferred_key = entry.get("model_key")
    if isinstance(preferred_key, str) and preferred_key.strip():
        normalized_key = preferred_key.strip()
        for model in models:
            if isinstance(model.key, str) and model.key.strip() == normalized_key:
                return model

    preferred_name = entry.get("model")
    if isinstance(preferred_name, str) and preferred_name.strip():
        normalized_name = preferred_name.strip()
        for model in models:
            if isinstance(model.name, str) and model.name.strip() == normalized_name:
                return model
    return None


def _choose_server_for_api(api_type: str, *, allow_back: bool) -> Optional[Dict[str, Any]]:
    servers = load_servers(api_type)
    servers = [server for server in servers if api_type in server.get("apis", {})]
    if not servers:
        return None

    preferred = _get_preferred_server(api_type, servers)
    if preferred is not None:
        return preferred

    selected = select_server(servers, allow_back=allow_back)
    if selected is None:
        return None
    _remember_server_selection(api_type, selected)
    return selected


_load_menu_state()


def _has_embedding_keyword(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(keyword in lowered for keyword in EMBED_MODEL_KEYWORDS)


def update_model_capabilities_from_metadata(name: str, metadata: Dict[str, Any]) -> None:
    info, changed = _ensure_model_entry(name)
    if info.get("is_embedding") and info.get("confirmed"):
        if changed:
            _save_model_capabilities()
        return

    detected = False
    if _has_embedding_keyword(name):
        detected = True

    details = metadata.get("details")
    if not detected and isinstance(details, dict):
        for key in ("model_type", "type", "family"):
            if _has_embedding_keyword(details.get(key)):
                detected = True
                break
        if not detected:
            families = details.get("families")
            if isinstance(families, list):
                for family in families:
                    if _has_embedding_keyword(family):
                        detected = True
                        break

    if not detected:
        model_info = metadata.get("model_info")
        if isinstance(model_info, dict):
            general = model_info.get("general")
            if isinstance(general, dict):
                for key in ("type", "category", "architecture"):
                    if _has_embedding_keyword(general.get(key)):
                        detected = True
                        break

    if detected:
        mark_model_as_embedding(name)
    elif changed:
        _save_model_capabilities()


def mark_model_as_embedding(name: str) -> None:
    info, changed = _ensure_model_entry(name)
    if not info.get("is_embedding"):
        info["is_embedding"] = True
        changed = True
    if not info.get("confirmed"):
        info["confirmed"] = True
        changed = True
    if "inferred" in info:
        info.pop("inferred")
        changed = True
    if changed:
        info["updated_at"] = _utcnow_iso()
        _save_model_capabilities()


def is_embedding_model(name: str) -> bool:
    info = MODEL_CAPABILITIES.get(name)
    if info and info.get("is_embedding"):
        return True
    if _has_embedding_keyword(name):
        info, changed = _ensure_model_entry(name)
        if not info.get("is_embedding"):
            info["is_embedding"] = True
            changed = True
        if not info.get("inferred"):
            info["inferred"] = True
            changed = True
        if changed:
            info["updated_at"] = _utcnow_iso()
            _save_model_capabilities()
        return True
    return False


def extract_error_message(response: Optional[requests.Response]) -> str:
    if response is None:
        return ""
    try:
        data = response.json()
        if isinstance(data, dict):
            for key in ("error", "message", "detail"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    except ValueError:
        pass

    text = getattr(response, "text", "") or ""
    text = text.strip()
    if len(text) > 200:
        text = text[:200] + "..."
    return text


def describe_http_error(error: requests.HTTPError) -> str:
    message = str(error)
    detail = extract_error_message(error.response)
    if detail:
        return f"{message} - {detail}"
    return message


def extract_embedding_vector(payload: Any) -> Optional[List[float]]:
    if isinstance(payload, dict):
        vector = payload.get("embedding")
        if isinstance(vector, list):
            return vector
        data = payload.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                inner = first.get("embedding")
                if isinstance(inner, list):
                    return inner
    return None


def format_embedding_response(vector: List[float]) -> str:
    if not vector:
        return "No embedding values returned by the server."
    dims = len(vector)
    preview_count = min(8, dims)
    preview = ", ".join(f"{value:.6f}" for value in vector[:preview_count])
    if dims > preview_count:
        preview += ", ..."
    return (
        "**Embedding generated**\n"
        f"- Dimensions: {dims}\n"
        f"- Preview: [{preview}]"
    )

def api_headers():
    return {"Content-Type": "application/json"}


def normalise_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def read_endpoints(api_type):
    candidates = []
    primary = CSV_PATHS.get(api_type)
    if primary:
        candidates.append(primary)
    if api_type == "ollama":
        candidates.append(LEGACY_CSV_PATH)

    for path in candidates:
        if not path or not path.exists():
            continue
        try:
            df = pd.read_csv(path, keep_default_na=False)
        except Exception as exc:
            print(f"{RED}Failed to read {path.name}: {exc}{RESET}")
            continue
        df = df.copy()
        if "api_type" in df.columns:
            df["api_type"] = df["api_type"].astype(str).str.lower()
            df = df[df["api_type"] == api_type]
        else:
            df["api_type"] = api_type
        if "port" in df.columns:
            df["port"] = pd.to_numeric(df["port"], errors="coerce")
            df = df.dropna(subset=["port"])
            df["port"] = df["port"].astype(int)
        if "ping" in df.columns:
            df["ping"] = pd.to_numeric(df["ping"], errors="coerce")
        return df

    columns = COLUMN_ORDER.get(api_type, BASE_COLUMNS)
    return pd.DataFrame(columns=columns)


def write_endpoints(api_type, df):
    path = CSV_PATHS.get(api_type)
    if not path:
        return
    out = df.copy()
    out["api_type"] = api_type
    if "port" in out.columns:
        out["port"] = pd.to_numeric(out["port"], errors="coerce").astype("Int64")
    if "ping" in out.columns:
        out["ping"] = pd.to_numeric(out["ping"], errors="coerce")
    columns = COLUMN_ORDER.get(api_type)
    if columns:
        for col in columns:
            if col not in out.columns:
                out[col] = ""
        ordered = columns + [c for c in out.columns if c not in columns]
        out = out[ordered]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def endpoint_mask(df, ip, port):
    if "ip" not in df.columns or "port" not in df.columns:
        return pd.Series(False, index=df.index)
    ports = pd.to_numeric(df["port"], errors="coerce").astype("Int64")
    return (df["ip"] == ip) & (ports == int(port))


def start_thinking_timer():
    start = time.time()
    stop_event = threading.Event()

    def updater():
        while not stop_event.is_set():
            elapsed = int(time.time() - start)
            print(
                f"\r\033[K{AI_COLOR}\U0001f5a5️ : Thinking... ({elapsed}s){RESET}",
                end="",
                flush=True,
            )
            time.sleep(1)

    threading.Thread(target=updater, daemon=True).start()
    return start, stop_event


def stop_thinking_timer(start, stop_event, timed_out=False):
    stop_event.set()
    elapsed = int(time.time() - start)
    status = " [Timed out]" if timed_out else ""
    print(
        f"\r\033[K{AI_COLOR}\U0001f5a5️ : Thinking... ({elapsed}s){status}{RESET}"
    )
    return elapsed


def get_input(prompt):
    """Read a line of input allowing basic arrow-key navigation."""
    if os.name == "nt":
        sys.stdout.write(prompt)
        sys.stdout.flush()
        buf = []
        pos = 0
        while True:
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                print()
                return "".join(buf).strip()
            if ch == "\x1b":
                print()
                return "ESC"
            if ch in ("\x00", "\xe0"):
                ch2 = msvcrt.getwch()
                if ch2 == "K":  # left
                    if pos > 0:
                        sys.stdout.write("\b")
                        pos -= 1
                elif ch2 == "M":  # right
                    if pos < len(buf):
                        sys.stdout.write(buf[pos])
                        pos += 1
                elif ch2 in ("H", "P"):
                    # ignore up/down
                    pass
                continue
            if ch == "\b":
                if pos > 0:
                    pos -= 1
                    del buf[pos]
                    sys.stdout.write("\b" + "".join(buf[pos:]) + " ")
                    sys.stdout.write("\b" * (len(buf) - pos + 1))
            else:
                buf.insert(pos, ch)
                sys.stdout.write("".join(buf[pos:]))
                pos += 1
                sys.stdout.write("\b" * (len(buf) - pos))
            sys.stdout.flush()
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            sys.stdout.write(prompt)
            sys.stdout.flush()
            buf = []
            pos = 0
            while True:
                ch = sys.stdin.read(1)
                if ch in ("\n", "\r"):
                    print()
                    return "".join(buf).strip()
                if ch == "\x7f":  # backspace
                    if pos > 0:
                        pos -= 1
                        del buf[pos]
                        sys.stdout.write("\b" + "".join(buf[pos:]) + " ")
                        sys.stdout.write("\b" * (len(buf) - pos + 1))
                    sys.stdout.flush()
                    continue
                if ch == "\x1b":
                    seq = _read_escape_sequence()
                    if seq.endswith("C"):  # right
                        if pos < len(buf):
                            sys.stdout.write(buf[pos])
                            pos += 1
                            sys.stdout.flush()
                        continue
                    if seq.endswith("D"):  # left
                        if pos > 0:
                            sys.stdout.write("\b")
                            pos -= 1
                            sys.stdout.flush()
                        continue
                    if seq.endswith(("A", "B")):
                        # ignore up/down
                        continue
                    print()
                    return "ESC"
                buf.insert(pos, ch)
                sys.stdout.write("".join(buf[pos:]))
                pos += 1
                sys.stdout.write("\b" * (len(buf) - pos))
                sys.stdout.flush()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)



def get_key():
    if os.name == "nt":
        while True:
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                return "ENTER"
            if ch == " ":
                return "SPACE"
            if ch in ("\x00", "\xe0"):
                ch2 = msvcrt.getwch()
                if ch2 == "H":
                    return "UP"
                if ch2 == "P":
                    return "DOWN"
                if ch2 == "M":
                    return "RIGHT"
                if ch2 == "K":
                    return "LEFT"
                continue
            if ch == "\x1b":
                return "ESC"
            return ch
    else:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = _read_escape_sequence()
                if seq.endswith("A"):
                    return "UP"
                if seq.endswith("B"):
                    return "DOWN"
                if seq.endswith("C"):
                    return "RIGHT"
                if seq.endswith("D"):
                    return "LEFT"
                return "ESC"
            if ch in ("\n", "\r"):
                return "ENTER"
            if ch == " ":
                return "SPACE"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


_MATRIX_ESCAPE_MAP = {
    "[A": "UP",
    "OA": "UP",
    "[B": "DOWN",
    "OB": "DOWN",
    "[5~": "PAGE_UP",
    "[6~": "PAGE_DOWN",
    "[H": "HOME",
    "OH": "HOME",
    "[F": "END",
    "OF": "END",
    "[1~": "HOME",
    "[4~": "END",
    "[7~": "HOME",
    "[8~": "END",
}


def interactive_menu(header, options, center=False):
    menu_options: List[tuple[str, Optional[str]]] = []
    for opt in options:
        if isinstance(opt, dict):
            label = str(opt.get("label", ""))
            style = opt.get("style") or opt.get("color")
        elif isinstance(opt, (list, tuple)):
            if not opt:
                label = ""
                style = None
            else:
                label = str(opt[0])
                style = opt[1] if len(opt) > 1 else None
        else:
            label = str(opt)
            style = None
        menu_options.append((label, style))

    if not menu_options:
        return None

    if DEBUG_MODE:
        print(f"{CYAN}{header}{RESET}")
        for idx, (label, _style) in enumerate(menu_options, 1):
            print(f"{GREEN}{idx}. {label}{RESET}")
        while True:
            choice = get_input(f"{CYAN}Choose an option: {RESET}")
            if choice == "ESC":
                return None
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(menu_options):
                    return index
            print(f"{RED}Invalid selection{RESET}")

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(f"{CYAN}{header}{RESET}")
        for idx, (label, _style) in enumerate(menu_options, 1):
            print(f"{GREEN}{idx}. {label}{RESET}")
        try:
            resp = input("Select option #: ").strip()
        except EOFError:
            return None
        return int(resp) - 1 if resp.isdigit() else None

    return _matrix_menu(header, menu_options, center=center)


class _MatrixMenuLayout(NamedTuple):
    visible_count: int
    instruction_text: str
    box_x: int
    box_y: int
    box_width: int
    box_height: int


def _calculate_matrix_layout(
    header_lines: List[str],
    options: List[tuple[str, Optional[str]]],
    center: bool,
    cols: int,
    rows: int,
) -> _MatrixMenuLayout:
    header_height = len(header_lines)
    blank_after_header = 1 if header_height else 0
    blank_before_instructions = 1
    instructions_lines = 1

    base_height = 2 + header_height + blank_after_header + blank_before_instructions + instructions_lines
    visible_count = min(len(options), max(1, rows - base_height))

    instruction_text = "↑/↓ Move  Enter Select  Esc Back"
    if len(options) > visible_count:
        instruction_text += "  PgUp/PgDn Scroll"

    max_header_width = max((len(line) for line in header_lines), default=0)
    max_option_width = max((len(label) for label, _ in options), default=0) + 4
    desired_content_width = max(max_header_width, max_option_width, len(instruction_text))
    max_allowed_width = max(4, cols - 2)
    desired_width = max(30, desired_content_width + 4)
    box_width = min(cols, max(6, min(max_allowed_width, desired_width)))

    box_height = min(rows, base_height + visible_count)
    box_x = max(1, (cols - box_width) // 2 + 1)
    box_y = max(1, (rows - box_height) // 2 + 1)

    return _MatrixMenuLayout(
        visible_count=visible_count,
        instruction_text=instruction_text,
        box_x=box_x,
        box_y=box_y,
        box_width=box_width,
        box_height=box_height,
    )


def _matrix_menu(
    header: str, options: List[tuple[str, Optional[str]]], *, center: bool
) -> Optional[int]:
    header_lines = header.splitlines() if header else []
    cols, rows = shutil.get_terminal_size(fallback=(80, 24))
    layout = _calculate_matrix_layout(header_lines, options, center, cols, rows)

    boxes: List[Dict[str, int]] = [
        {
            "top": layout.box_y,
            "bottom": layout.box_y + layout.box_height - 1,
            "left": layout.box_x,
            "right": layout.box_x + layout.box_width - 1,
        }
    ]

    stop_event = threading.Event()
    thread = threading.Thread(
        target=rain,
        kwargs={
            "persistent": True,
            "stop_event": stop_event,
            "boxes": boxes,
            "clear_screen": False,
        },
        daemon=True,
    )
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
    thread.start()

    try:
        return _matrix_menu_loop(header_lines, options, boxes, center=center)
    finally:
        stop_event.set()
        thread.join()


def _matrix_menu_loop(
    header_lines: List[str],
    options: List[tuple[str, Optional[str]]],
    boxes: List[Dict[str, int]],
    *,
    center: bool,
) -> Optional[int]:
    selected = 0
    offset = 0

    fd = None
    old_settings = None
    if os.name != "nt":
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    try:
        while True:
            cols, rows = shutil.get_terminal_size(fallback=(80, 24))
            if not options:
                return None

            layout = _calculate_matrix_layout(header_lines, options, center, cols, rows)
            visible_count = layout.visible_count
            instruction_text = layout.instruction_text
            box_width = layout.box_width
            box_height = layout.box_height
            box_x = layout.box_x
            box_y = layout.box_y

            max_offset = max(0, len(options) - visible_count)
            if offset > max_offset:
                offset = max_offset
            if selected >= len(options):
                selected = max(0, len(options) - 1)
            if selected < offset:
                offset = selected
            elif selected >= offset + visible_count:
                offset = max(0, selected - visible_count + 1)

            if boxes:
                boxes[0].update(
                    {
                        "top": box_y,
                        "bottom": box_y + box_height - 1,
                        "left": box_x,
                        "right": box_x + box_width - 1,
                    }
                )
            else:
                boxes.append(
                    {
                        "top": box_y,
                        "bottom": box_y + box_height - 1,
                        "left": box_x,
                        "right": box_x + box_width - 1,
                    }
                )

            _render_matrix_menu(
                header_lines,
                options,
                offset,
                visible_count,
                selected,
                instruction_text,
                box_x,
                box_y,
                box_width,
                box_height,
                center,
            )

            action = _read_matrix_action(fd)
            if action == "UP":
                selected = (selected - 1) % len(options)
            elif action == "DOWN":
                selected = (selected + 1) % len(options)
            elif action == "PAGE_UP":
                selected = max(0, selected - visible_count)
            elif action == "PAGE_DOWN":
                selected = min(len(options) - 1, selected + visible_count)
            elif action == "HOME":
                selected = 0
            elif action == "END":
                selected = len(options) - 1
            elif action == "ENTER":
                return selected
            elif action == "ESC":
                return None

            if selected < offset:
                offset = selected
            elif selected >= offset + visible_count:
                offset = max(0, selected - visible_count + 1)
    finally:
        if os.name != "nt" and old_settings is not None and fd is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _render_matrix_menu(
    header_lines: List[str],
    options: List[tuple[str, Optional[str]]],
    offset: int,
    visible_count: int,
    selected: int,
    instruction_text: str,
    box_x: int,
    box_y: int,
    box_width: int,
    box_height: int,
    center: bool,
) -> None:
    buffer: List[str] = []
    top_border = f"{GREEN}┌{'─' * (box_width - 2)}┐{RESET}" if box_width >= 2 else ""
    bottom_border = f"{GREEN}└{'─' * (box_width - 2)}┘{RESET}" if box_width >= 2 else ""
    if top_border:
        buffer.append(f"\033[{box_y};{box_x}H{top_border}")
    inner_width = max(0, box_width - 2)
    for row in range(1, box_height - 1):
        buffer.append(
            f"\033[{box_y + row};{box_x}H{GREEN}│{RESET}{' ' * inner_width}{GREEN}│{RESET}"
        )
    if bottom_border:
        buffer.append(f"\033[{box_y + box_height - 1};{box_x}H{bottom_border}")

    inner_left = box_x + 1
    inner_width = max(1, box_width - 2)
    content_left = inner_left + 1 if inner_width > 2 else inner_left
    content_width = max(1, inner_width - 2) if inner_width > 2 else inner_width
    line_y = box_y + 1

    if header_lines:
        for line in header_lines:
            buffer.append(f"\033[{line_y};{inner_left}H{' ' * inner_width}")
            truncated = line[: content_width if not center else inner_width]
            if center:
                start_x = inner_left + max((inner_width - len(truncated)) // 2, 0)
            else:
                start_x = content_left
                truncated = truncated[:content_width]
            buffer.append(f"\033[{line_y};{start_x}H{BOLD}{GREEN}{truncated}{RESET}")
            line_y += 1
        buffer.append(f"\033[{line_y};{inner_left}H{' ' * inner_width}")
        line_y += 1

    options_start_line = line_y
    visible_options = options[offset : offset + visible_count]
    color_map = {
        None: GREEN,
        "default": GREEN,
        "warning": WARNING,
        "danger": RED,
    }
    for index, (label, style) in enumerate(visible_options):
        actual = offset + index
        marker = "▶" if actual == selected else " "
        option_text = f"{marker} {label}" if content_width > 2 else label
        option_text = option_text[:content_width]
        buffer.append(f"\033[{line_y};{inner_left}H{' ' * inner_width}")
        color = CYAN if actual == selected else color_map.get(style, GREEN)
        emphasis = BOLD if actual == selected else ""
        buffer.append(
            f"\033[{line_y};{content_left}H{emphasis}{color}{option_text}{RESET}"
        )
        line_y += 1

    scroll_column = box_x + box_width - 2
    if offset > 0 and scroll_column >= box_x:
        buffer.append(f"\033[{options_start_line};{scroll_column}H{CYAN}▲{RESET}")
    if offset + visible_count < len(options) and scroll_column >= box_x:
        buffer.append(
            f"\033[{options_start_line + visible_count - 1};{scroll_column}H{CYAN}▼{RESET}"
        )

    buffer.append(f"\033[{line_y};{inner_left}H{' ' * inner_width}")
    line_y += 1

    if center:
        instruction_line = instruction_text[: inner_width]
        start_x = inner_left + max((inner_width - len(instruction_line)) // 2, 0)
    else:
        instruction_line = instruction_text[: content_width]
        start_x = content_left
    buffer.append(f"\033[{line_y};{inner_left}H{' ' * inner_width}")
    buffer.append(f"\033[{line_y};{start_x}H{CYAN}{instruction_line}{RESET}")

    sys.stdout.write("".join(buffer))
    sys.stdout.flush()


def _read_matrix_action(fd: Optional[int]) -> Optional[str]:
    try:
        if os.name == "nt":
            while True:
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    return "ENTER"
                if ch == " ":
                    return "ENTER"
                if ch == "\x1b":
                    return "ESC"
                if ch in ("\x00", "\xe0"):
                    code = msvcrt.getwch()
                    mapping = {
                        "H": "UP",
                        "P": "DOWN",
                        "I": "PAGE_UP",
                        "Q": "PAGE_DOWN",
                        "G": "HOME",
                        "O": "END",
                        "K": "LEFT",
                        "M": "RIGHT",
                    }
                    action = mapping.get(code)
                    if action in ("LEFT", "RIGHT"):
                        continue
                    if action:
                        return action
                    continue
                if ch in ("k", "K", "w", "W"):
                    return "UP"
                if ch in ("j", "J", "s", "S"):
                    return "DOWN"
                if ch == "\t":
                    return "ENTER"
                if ch == "\x03":
                    raise KeyboardInterrupt
                # Ignore other keys
        else:
            ch = sys.stdin.read(1)
            if not ch:
                return None
            if ch in ("\r", "\n"):
                return "ENTER"
            if ch == " ":
                return "ENTER"
            if ch == "\x1b":
                seq = _read_escape_sequence()
                if not seq:
                    return "ESC"
                action = _MATRIX_ESCAPE_MAP.get(seq)
                if action:
                    return action
                return None
            if ch in ("k", "K", "w", "W"):
                return "UP"
            if ch in ("j", "J", "s", "S"):
                return "DOWN"
            if ch == "\x7f":
                return None
            if ch == "\x03":
                raise KeyboardInterrupt
        return None
    except KeyboardInterrupt:
        raise


if os.name != "nt":
    from curses_nav import get_input as curses_get_input
    get_input = curses_get_input


def _display_with_chafapy(image_path: Path) -> bool:
    if chafa is None or PILImage is None:
        return False

    try:
        with PILImage.open(image_path) as pil_image:
            pil_image = pil_image.convert("RGBA")
            width, height = pil_image.size
            pixels = pil_image.tobytes()
    except Exception as exc:  # pragma: no cover - depends on runtime image data
        print(f"{YELLOW}Failed to load image preview: {exc}. Image saved to {image_path}{RESET}")
        return False

    try:
        term_size = shutil.get_terminal_size(fallback=(80, 24))
        config = chafa.CanvasConfig()
        config.pixel_mode = chafa.PixelMode.CHAFA_PIXEL_MODE_SYMBOLS
        config.width = max(1, term_size.columns)
        config.height = max(1, term_size.lines - 2)
        try:
            config.calc_canvas_geometry(width, height, font_ratio=0.5, zoom=True)
        except Exception:  # pragma: no cover - depends on chafa implementation
            pass

        canvas = chafa.Canvas(config)
        canvas.draw_all_pixels(
            chafa.PixelType.CHAFA_PIXEL_RGBA8_UNASSOCIATED,
            pixels,
            width,
            height,
            width * 4,
        )
        output = canvas.print(fallback=True)
        sys.stdout.write(output.decode("utf-8", errors="ignore"))
        if not output.endswith(b"\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        return True
    except Exception as exc:  # pragma: no cover - depends on terminal capabilities
        print(
            f"{YELLOW}Failed to render image preview with chafa.py: {exc}. Falling back to chafa CLI if available.{RESET}"
        )
        return False


def _display_with_chafa_cli(image_path: Path) -> None:
    preferred_cli = "chafa.py"
    fallback_cli = "chafa"

    for executable, label in ((preferred_cli, "chafa.py"), (fallback_cli, "chafa")):
        try:
            result = subprocess.run([executable, str(image_path)], check=False)
        except FileNotFoundError:
            continue

        if result.returncode in (0, 1):
            if executable == fallback_cli:
                print(
                    f"{YELLOW}Used legacy '{label}' binary. Consider installing the Python port from https://github.com/GuardKenzie/chafa.py for improved compatibility (MagickWand required for Loader).{RESET}"
                )
            return

        print(
            f"{YELLOW}{label} returned code {result.returncode}. Image saved to {image_path}{RESET}"
        )
        return

    print(
        f"{YELLOW}chafa.py CLI not installed. Install it from https://github.com/GuardKenzie/chafa.py (requires MagickWand) or add the 'Pillow' package to enable ANSI previews.{RESET}"
    )


def _display_with_ansi(path: Path) -> bool:
    if PILImage is None:
        return False

    try:
        with PILImage.open(path) as pil_image:
            pil_image = pil_image.convert("RGB")
            term_size = shutil.get_terminal_size(fallback=(80, 24))
            max_width = max(2, min(term_size.columns, 160))
            width, height = pil_image.size
            if width == 0 or height == 0:
                return False

            aspect_ratio = height / width
            new_height = max(1, int(aspect_ratio * max_width * 0.5))
            pil_image = pil_image.resize((max_width, new_height))

            pixels = list(pil_image.getdata())

        for row_index in range(new_height):
            start = row_index * max_width
            end = start + max_width
            row_pixels = pixels[start:end]
            row = "".join(f"\033[48;2;{r};{g};{b}m " for r, g, b in row_pixels)
            print(f"{row}{RESET}")
        return True
    except Exception as exc:  # pragma: no cover - depends on runtime image data
        print(
            f"{YELLOW}Failed to render preview using Pillow ANSI fallback: {exc}. Image saved to {path}{RESET}"
        )
        return False


def display_with_chafa(image_path):
    path = Path(image_path)

    if os.name == "nt":
        if _display_with_ansi(path):
            return
        print(f"{YELLOW}Preview not supported on Windows. Image saved to {image_path}{RESET}")
        return

    if _display_with_chafapy(path):
        return

    if _display_with_ansi(path):
        return

    _display_with_chafa_cli(path)


def _cleanup_image_result(result, *, discard: bool = False):
    if not isinstance(result, dict):
        return

    cached = bool(result.get("cached"))
    if cached and not discard:
        return

    path = result.get("path")
    metadata_path = result.get("metadata_path")

    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception as exc:
            print(f"{YELLOW}Failed to remove preview image {path}: {exc}{RESET}")
    if metadata_path:
        try:
            Path(metadata_path).unlink(missing_ok=True)
        except Exception:
            pass


def _sanitize_filename(candidate: str) -> str:
    """Return a filesystem-friendly version of ``candidate``."""

    if not isinstance(candidate, str):
        return ""
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", candidate.strip())
    return sanitized.strip("._")


def _resolve_unique_path(path: Path) -> Path:
    """Return ``path`` or a variant with a numeric suffix that does not exist."""

    candidate = path
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        counter += 1
    return candidate


def _print_board_image_summary(metadata: Dict[str, Any]) -> None:
    prompt = metadata.get("prompt") if isinstance(metadata, dict) else None
    negative = metadata.get("negative_prompt") if isinstance(metadata, dict) else None
    scheduler = metadata.get("scheduler") if isinstance(metadata, dict) else None
    seed = metadata.get("seed") if isinstance(metadata, dict) else None
    model_info = metadata.get("model") if isinstance(metadata, dict) else None

    if isinstance(prompt, str) and prompt.strip():
        print(f"{CYAN}Prompt:{RESET} {prompt.strip()}")
    if isinstance(negative, str) and negative.strip():
        print(f"{CYAN}Negative:{RESET} {negative.strip()}")
    if isinstance(model_info, dict):
        model_name = model_info.get("name") or model_info.get("model")
        model_base = model_info.get("base") or model_info.get("base_model")
        details = []
        if isinstance(model_name, str) and model_name.strip():
            details.append(model_name.strip())
        if isinstance(model_base, str) and model_base.strip():
            details.append(model_base.strip())
        if details:
            print(f"{CYAN}Model:{RESET} {' • '.join(details)}")
    if scheduler:
        print(f"{CYAN}Scheduler:{RESET} {scheduler}")
    if seed is not None:
        print(f"{CYAN}Seed:{RESET} {seed}")


def _coerce_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and math.isfinite(value):
        if isinstance(value, float):
            return int(value)
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except (ValueError, TypeError):
            return default
    return default


def _print_invoke_batch_progress(status: Dict[str, Any]) -> None:
    total = _coerce_int(status.get("total"), default=0)
    completed = _coerce_int(status.get("completed"), default=0)
    processing = _coerce_int(status.get("processing"), default=0)
    pending = _coerce_int(status.get("pending"), default=0)
    status_label = status.get("status") if isinstance(status.get("status"), str) else ""
    status_text = status_label.strip().title() if status_label else "Unknown"

    if total > 0:
        progress_bits = [f"{completed}/{total} complete"]
    else:
        progress_bits = [f"{completed} complete"]

    details: List[str] = []
    if processing > 0:
        details.append(f"{processing} processing")
    if pending > 0:
        details.append(f"{pending} pending")

    if details:
        progress_bits.append("(" + ", ".join(details) + ")")

    print(f"{CYAN}Batch status:{RESET} {status_text} — {' '.join(progress_bits)}")


def _is_batch_complete(status: Dict[str, Any]) -> bool:
    status_label = status.get("status") if isinstance(status.get("status"), str) else ""
    normalized = status_label.strip().lower()
    if normalized in {"completed", "failed"}:
        return True

    total = _coerce_int(status.get("total"), default=0)
    completed = _coerce_int(status.get("completed"), default=0)
    if total > 0 and completed >= total:
        return True

    pending = _coerce_int(status.get("pending"), default=0)
    processing = _coerce_int(status.get("processing"), default=0)
    if total == 0 and pending == 0 and processing == 0 and completed > 0:
        return True

    return False


def _poll_invoke_batch_status(
    client: InvokeAIClient,
    batch_id: str,
    *,
    timeout: float = BATCH_STATUS_TIMEOUT,
    poll_interval: float = BATCH_STATUS_POLL_INTERVAL,
) -> Optional[Dict[str, Any]]:
    start = time.monotonic()
    last_status: Optional[Dict[str, Any]] = None

    while True:
        now = time.monotonic()
        if now - start > timeout:
            print(
                f"{YELLOW}Timed out waiting for batch {batch_id} to complete. Check the {TERMINALAI_BOARD_NAME} board for results.{RESET}"
            )
            return last_status

        try:
            status = client.get_batch_status(
                batch_id,
                include_preview=True,
                board_name=TERMINALAI_BOARD_NAME,
            )
        except InvokeAIClientError as exc:
            print(f"{YELLOW}Failed to poll batch {batch_id}: {exc}{RESET}")
            return last_status
        except requests.RequestException as exc:
            print(f"{YELLOW}Network error while polling batch {batch_id}: {exc}{RESET}")
            return last_status

        if not isinstance(status, dict):
            print(
                f"{YELLOW}InvokeAI returned an unexpected batch status payload. Check the {TERMINALAI_BOARD_NAME} board for updates.{RESET}"
            )
            return last_status

        last_status = status
        _print_invoke_batch_progress(status)

        preview = status.get("preview")
        if isinstance(preview, dict):
            return status

        if _is_batch_complete(status):
            return status

        if poll_interval > 0:
            time.sleep(poll_interval)


def _present_invoke_result(result: Dict[str, Any], *, label: str = "Batch preview available") -> None:
    if not isinstance(result, dict):
        return

    print(f"{GREEN}{label}:{RESET}")
    path_value = result.get("path")
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}

    if path_value:
        print(f"{CYAN}Preview saved to {path_value}{RESET}")

    if metadata:
        _print_board_image_summary(metadata)

    if path_value:
        display_with_chafa(path_value)

    _cleanup_image_result(result, discard=True)


def _load_invoke_final_result(
    client: InvokeAIClient,
    status: Dict[str, Any],
    batch_id: str,
) -> Optional[Dict[str, Any]]:
    try:
        board_id = client.ensure_board(TERMINALAI_BOARD_NAME)
    except InvokeAIClientError as exc:
        print(f"{YELLOW}Unable to load final image: {exc}{RESET}")
        return None
    except requests.RequestException as exc:
        print(f"{YELLOW}Network error while loading final image: {exc}{RESET}")
        return None

    if not _is_valid_terminalai_board_id(board_id):
        print(f"{YELLOW}{TERMINALAI_BOARD_RESOLUTION_ERROR}.{RESET}")
        return None

    try:
        board_entries = client.list_board_images(board_id, limit=10)
    except InvokeAIClientError as exc:
        print(f"{YELLOW}Unable to load final image from board: {exc}{RESET}")
        return None
    except requests.RequestException as exc:
        print(f"{YELLOW}Network error while loading final image from board: {exc}{RESET}")
        return None

    if not board_entries:
        return None

    identifiers: set[str] = set()

    def _add_identifier(value: Any) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                identifiers.add(text)
        elif isinstance(value, (int, float)) and math.isfinite(value):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            identifiers.add(str(value).strip())

    _add_identifier(batch_id)
    queue_item_id = status.get("queue_item_id") if isinstance(status, dict) else None
    if queue_item_id is not None:
        _add_identifier(queue_item_id)

    queue_items = status.get("queue_items") if isinstance(status, dict) else None
    if isinstance(queue_items, list):
        for entry in queue_items:
            if isinstance(entry, dict):
                _add_identifier(entry.get("item_id"))

    def _gather_identifiers(value: Any, seen: set[int]) -> Iterable[str]:
        if isinstance(value, dict):
            obj_id = id(value)
            if obj_id in seen:
                return []
            seen.add(obj_id)
            keys = (
                "batch_id",
                "batchId",
                "batchID",
                "item_id",
                "queue_item_id",
                "queue_item",
                "queueId",
                "session_id",
            )
            results: List[str] = []
            for key in keys:
                candidate = value.get(key)
                if isinstance(candidate, str):
                    candidate_text = candidate.strip()
                    if candidate_text:
                        results.append(candidate_text)
                elif isinstance(candidate, (int, float)) and math.isfinite(candidate):
                    if isinstance(candidate, float) and candidate.is_integer():
                        candidate = int(candidate)
                    results.append(str(candidate).strip())
            for child in value.values():
                results.extend(_gather_identifiers(child, seen))
            return results
        if isinstance(value, list):
            obj_id = id(value)
            if obj_id in seen:
                return []
            seen.add(obj_id)
            results: List[str] = []
            for item in value:
                results.extend(_gather_identifiers(item, seen))
            return results
        return []

    def _matches(entry: Dict[str, Any]) -> bool:
        if not identifiers:
            return False
        entry_identifiers = set(_gather_identifiers(entry, set()))
        return any(identifier in entry_identifiers for identifier in identifiers)

    for candidate in board_entries:
        if not isinstance(candidate, dict):
            continue
        if _matches(candidate):
            try:
                return client.retrieve_board_image(
                    image_info=candidate,
                    board_name=TERMINALAI_BOARD_NAME,
                )
            except InvokeAIClientError as exc:
                print(f"{YELLOW}Failed to download final image: {exc}{RESET}")
                return None
            except requests.RequestException as exc:
                print(f"{YELLOW}Network error while downloading final image: {exc}{RESET}")
                return None

    if len(board_entries) == 1:
        fallback_entry = board_entries[0]
    else:
        fallback_entry = board_entries[0] if isinstance(board_entries[0], dict) else None

    if isinstance(fallback_entry, dict):
        try:
            return client.retrieve_board_image(
                image_info=fallback_entry,
                board_name=TERMINALAI_BOARD_NAME,
            )
        except InvokeAIClientError as exc:
            print(f"{YELLOW}Failed to download final image: {exc}{RESET}")
        except requests.RequestException as exc:
            print(f"{YELLOW}Network error while downloading final image: {exc}{RESET}")

    return None


def _present_invoke_preview(preview: Dict[str, Any]) -> None:
    """Backward compatible wrapper for legacy preview presenter."""

    _present_invoke_result(preview)


def confirm_exit():
    choice = interactive_menu("Exit?", ["No", "Yes"])
    return choice == 1


def _browse_board_images(client: InvokeAIClient, board: Dict[str, Any]) -> None:
    board_name_value = board.get("name") if isinstance(board, dict) else None
    board_name = (
        board_name_value.strip()
        if isinstance(board_name_value, str) and board_name_value.strip()
        else board_name_value
    ) or "Board"

    board_id_value = board.get("id") if isinstance(board, dict) else None
    board_id = _normalize_board_id(board_id_value)
    if not board_id and isinstance(board, dict) and board.get("is_uncategorized"):
        board_id = UNCATEGORIZED_BOARD_ID

    if (
        board_name == TERMINALAI_BOARD_NAME
        and not _is_valid_terminalai_board_id(board_id)
    ):
        print(f"{RED}{TERMINALAI_BOARD_ID_ERROR_MESSAGE}{RESET}")
        get_input(f"{CYAN}Press Enter to return{RESET}")
        return

    if not board_id:
        print(f"{RED}Selected board entry is missing an id.{RESET}")
        get_input(f"{CYAN}Press Enter to return{RESET}")
        return

    page_size = 5
    entries: List[Dict[str, Any]] = []
    offset = 0
    loaded_all = False
    index = 0

    def _fetch_next_batch(*, interactive: bool = True) -> bool:
        nonlocal offset, loaded_all, entries
        if loaded_all:
            return False
        try:
            batch = client.list_board_images(board_id, limit=page_size, offset=offset)
        except InvokeAIClientError as exc:
            print(f"{RED}{exc}{RESET}")
            if interactive:
                get_input(f"{CYAN}Press Enter to return{RESET}")
            return False
        except requests.RequestException as exc:
            print(f"{RED}Failed to retrieve board images: {exc}{RESET}")
            if interactive:
                get_input(f"{CYAN}Press Enter to return{RESET}")
            return False

        if not batch:
            loaded_all = True
            return False

        entries.extend(batch)
        offset += len(batch)
        if len(batch) < page_size:
            loaded_all = True
        return True

    if not _fetch_next_batch() and not entries:
        print(f"{YELLOW}No images found on board {board_name}.{RESET}")
        print(
            f"{CYAN}Press Esc to return or \u2192/D to refresh.{RESET}"
        )
        while True:
            key = get_key()
            if key in ("d", "D"):
                key = "RIGHT"
            if key == "ESC":
                return
            if key == "RIGHT":
                print(f"{CYAN}Refreshing board images...{RESET}")
                offset = 0
                loaded_all = False
                if _fetch_next_batch() or entries:
                    break
                print(f"{YELLOW}No images found on board {board_name}.{RESET}")
                print(
                    f"{CYAN}Press Esc to return or \u2192/D to refresh.{RESET}"
                )

    while entries:
        if index < 0:
            index = 0
        if index >= len(entries):
            index = len(entries) - 1

        entry = entries[index]
        image_name_value = entry.get("image_name") if isinstance(entry, dict) else None
        result: Optional[Dict[str, Any]] = None

        cache_lookup = getattr(client, "get_cached_image_result", None)
        if (
            callable(cache_lookup)
            and isinstance(image_name_value, str)
            and image_name_value.strip()
        ):
            try:
                cached_candidate = cache_lookup(image_name_value)
            except Exception:
                cached_candidate = None
            if isinstance(cached_candidate, dict):
                result = cached_candidate

        if result is None:
            try:
                result = client.retrieve_board_image(image_info=entry, board_name=board_name)
            except InvokeAIClientError as exc:
                print(f"{YELLOW}Skipping image: {exc}{RESET}")
                entries.pop(index)
                if not entries:
                    break
                if index >= len(entries):
                    index = len(entries) - 1
                continue
            except requests.RequestException as exc:
                print(f"{YELLOW}Failed to download image: {exc}{RESET}")
                entries.pop(index)
                if not entries:
                    break
                if index >= len(entries):
                    index = len(entries) - 1
                continue

        clear_screen()
        _print_board_view_header(client, board_name)
        print(f"{CYAN}Image {index + 1} of {len(entries)}{RESET}")

        path_value = result.get("path") if isinstance(result, dict) else None
        metadata = result.get("metadata") if isinstance(result, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        if not metadata and isinstance(entry, dict):
            entry_metadata = entry.get("metadata")
            if isinstance(entry_metadata, dict):
                metadata = entry_metadata
                if isinstance(result, dict):
                    result = dict(result)
                    result["metadata"] = metadata
        if path_value:
            print(f"{CYAN}Preview saved to {path_value}{RESET}")
        if isinstance(metadata, dict):
            _print_board_image_summary(metadata)
        if path_value:
            display_with_chafa(path_value)

        current_path = Path(path_value) if path_value else None
        metadata_path_value = result.get("metadata_path") if isinstance(result, dict) else None

        while True:
            print(
                f"{CYAN}\u2190/A for previous, \u2192/D for next, Enter to save, X to discard cache, Esc to return.{RESET}"
            )
            action = get_key()
            if action in ("a", "A"):
                action = "LEFT"
            elif action in ("d", "D"):
                action = "RIGHT"

            if action in ("x", "X"):
                if not current_path:
                    print(f"{YELLOW}No cached preview available to discard.{RESET}")
                    continue
                _cleanup_image_result(result, discard=True)
                print(f"{GREEN}Cached preview removed.{RESET}")
                current_path = None
                metadata_path_value = None
                if isinstance(result, dict):
                    result = dict(result)
                    result["path"] = None
                    result["metadata_path"] = None
                    result["cached"] = False
                continue

            if action == "ENTER":
                if not current_path:
                    print(f"{YELLOW}No preview available to save.{RESET}")
                    continue

                default_name = None
                prompt_value = metadata.get("prompt") if isinstance(metadata, dict) else None
                if isinstance(prompt_value, str) and prompt_value.strip():
                    default_name = prompt_value.strip().splitlines()[0]
                    if len(default_name) > 80:
                        default_name = default_name[:80].rstrip() + "\u2026"

                suggested = _sanitize_filename(default_name) if default_name else ""
                fallback = current_path.stem
                prompt_text = (
                    f"{CYAN}Save image as [{suggested or fallback}]: {RESET}"
                )
                response = get_input(prompt_text)
                if response == "ESC":
                    continue
                final_name = response.strip() or default_name or fallback
                sanitized = _sanitize_filename(final_name)
                if not sanitized:
                    sanitized = fallback

                target_path = current_path.with_name(f"{sanitized}{current_path.suffix}")
                if target_path != current_path:
                    target_path = _resolve_unique_path(target_path)
                    try:
                        current_path.rename(target_path)
                        print(f"{GREEN}Image saved to {target_path}{RESET}")
                        current_path = target_path
                        if isinstance(result, dict):
                            result = dict(result)
                            result["path"] = str(target_path)
                    except OSError as exc:
                        print(f"{YELLOW}Failed to rename image: {exc}{RESET}")
                else:
                    print(f"{GREEN}Image retained at {current_path}{RESET}")

                if metadata_path_value:
                    meta_path = Path(metadata_path_value)
                    target_meta = meta_path.with_name(f"{current_path.stem}{meta_path.suffix}")
                    if target_meta != meta_path:
                        target_meta = _resolve_unique_path(target_meta)
                        try:
                            meta_path.rename(target_meta)
                            metadata_path_value = str(target_meta)
                            if isinstance(result, dict):
                                result = dict(result)
                                result["metadata_path"] = str(target_meta)
                        except OSError as exc:
                            print(f"{YELLOW}Failed to rename metadata file: {exc}{RESET}")
                continue

            if action == "RIGHT":
                if index == len(entries) - 1:
                    if not loaded_all:
                        if not _fetch_next_batch(interactive=False):
                            if loaded_all:
                                print(
                                    f"{YELLOW}Already viewing the newest image on this board.{RESET}"
                                )
                            else:
                                print(
                                    f"{YELLOW}Unable to load additional images for this board right now.{RESET}"
                                )
                            continue
                    else:
                        print(f"{YELLOW}Already viewing the newest image on this board.{RESET}")
                        continue
                print(f"{CYAN}Loading next image...{RESET}")
                index += 1
                break

            if action == "LEFT":
                if index == 0:
                    print(f"{YELLOW}Already viewing the oldest image loaded for this board.{RESET}")
                    continue
                print(f"{CYAN}Loading previous image...{RESET}")
                index -= 1
                break

            if action == "ESC":
                return

        # end of inner loop

    if not entries:
        print(f"{YELLOW}No images remain on board {board_name}.{RESET}")
        get_input(f"{CYAN}Press Enter to return{RESET}")


def _view_server_boards(client: InvokeAIClient) -> None:
    try:
        boards = client.list_boards()
    except InvokeAIClientError as exc:
        print(f"{RED}{exc}{RESET}")
        get_input(f"{CYAN}Press Enter to return{RESET}")
        return
    except requests.RequestException as exc:
        print(f"{RED}Failed to fetch boards: {exc}{RESET}")
        get_input(f"{CYAN}Press Enter to return{RESET}")
        return

    if not boards:
        print(f"{YELLOW}No boards were reported by the server.{RESET}")
        get_input(f"{CYAN}Press Enter to return{RESET}")
        return

    while True:
        options = []
        for board in boards:
            label = board.get("name", "Unnamed board")
            count = board.get("count")
            if isinstance(count, int):
                label = f"{label} ({count})"
            options.append(label)
        options.append("[Back]")

        choice = interactive_menu(f"Boards on {client.nickname}:", options)
        if choice is None or choice == len(options) - 1:
            return

        board = boards[choice]
        clear_screen()
        print(f"{BOLD}{CYAN}Browsing board: {board.get('name', 'Unnamed')}{RESET}")
        _browse_board_images(client, board)
        clear_screen()


def _check_batch_status(client: InvokeAIClient) -> None:
    while True:
        batch_id = get_input(f"{CYAN}Batch id (ESC to cancel): {RESET}")
        if batch_id == "ESC":
            return
        batch_id = batch_id.strip()
        if not batch_id:
            print(f"{RED}Batch id cannot be empty{RESET}")
            continue

        try:
            status = client.get_batch_status(
                batch_id,
                include_preview=True,
                board_name=TERMINALAI_BOARD_NAME,
            )
        except InvokeAIClientError as exc:
            print(f"{RED}{exc}{RESET}")
            get_input(f"{CYAN}Press Enter to try another batch id{RESET}")
            return
        except requests.RequestException as exc:
            print(f"{RED}Network error: {exc}{RESET}")
            get_input(f"{CYAN}Press Enter to try another batch id{RESET}")
            return

        print(f"{GREEN}Batch {batch_id} status: {status.get('status', 'unknown').title()}{RESET}")

        total = status.get("total") or 0
        completed = status.get("completed") or 0
        processing = status.get("processing") or 0
        pending = status.get("pending") or 0
        failed = status.get("failed") or 0
        print(f"{CYAN}Progress:{RESET} {completed}/{total} complete")
        if processing:
            print(f"{CYAN}Processing:{RESET} {processing}")
        if pending:
            print(f"{CYAN}Waiting in queue:{RESET} {pending}")
        if failed:
            print(f"{YELLOW}Failed items in batch:{RESET} {failed}")

        queue_item_id = status.get("queue_item_id")
        if queue_item_id:
            print(f"{CYAN}Queue item id:{RESET} {queue_item_id}")

        queue_items = status.get("queue_items")
        if isinstance(queue_items, list) and len(queue_items) > 1:
            print(f"{CYAN}Queue entries:{RESET}")
            for entry in queue_items:
                item_label = entry.get("item_id") or "(unknown)"
                entry_status = entry.get("status") or "unknown"
                print(f"  - {item_label}: {entry_status}")

        eta_seconds = status.get("eta_seconds")
        if isinstance(eta_seconds, (int, float)) and eta_seconds > 0:
            eta_delta = datetime.timedelta(seconds=int(eta_seconds))
            print(f"{CYAN}Estimated wait:{RESET} ~{eta_delta}")

        preview = status.get("preview") if isinstance(status, dict) else None
        preview_error = status.get("preview_error") if isinstance(status, dict) else None

        if isinstance(preview, dict):
            path_value = preview.get("path")
            metadata = preview.get("metadata") if isinstance(preview.get("metadata"), dict) else {}
            if path_value:
                print(f"{CYAN}Preview saved to {path_value}{RESET}")
            if metadata:
                _print_board_image_summary(metadata)
            if path_value:
                display_with_chafa(path_value)
            get_input(f"{CYAN}Press Enter to return to the InvokeAI menu{RESET}")
            _cleanup_image_result(preview, discard=True)
            return

        if preview_error:
            print(f"{YELLOW}Preview unavailable: {preview_error}{RESET}")
        else:
            print(f"{YELLOW}Preview not available yet. The batch is still processing.{RESET}")

        get_input(f"{CYAN}Press Enter to return to the InvokeAI menu{RESET}")
        return


def _run_generation_flow(client: InvokeAIClient, models: List[InvokeAIModel]) -> None:
    if not models:
        print(f"{RED}No InvokeAI models reported by this server{RESET}")
        get_input(f"{CYAN}Press Enter to return{RESET}")
        return

    try:
        scheduler_options = client.list_schedulers()
    except InvokeAIClientError as exc:
        print(f"{YELLOW}Failed to retrieve scheduler list: {exc}{RESET}")
        scheduler_options = []
    except requests.RequestException as exc:
        print(f"{YELLOW}Failed to retrieve scheduler list: {exc}{RESET}")
        scheduler_options = []

    if not scheduler_options:
        scheduler_options = [DEFAULT_SCHEDULER]

    while True:
        model = _get_preferred_invoke_model(models)
        if model is None:
            model = select_invoke_model(models)
            if model is None:
                return
        _remember_model_selection("invokeai", model.name, model.key)

        refresh_prompt_view = True
        while True:
            if refresh_prompt_view:
                clear_screen()
                _print_invoke_prompt_header(client, model)
            refresh_prompt_view = True

            prompt = get_input(f"{CYAN}Prompt: {RESET}")
            if prompt == "ESC":
                break
            if not prompt.strip():
                print(f"{RED}Prompt cannot be empty{RESET}")
                refresh_prompt_view = False
                continue

            negative = get_input(f"{CYAN}Negative prompt (optional): {RESET}")
            if negative == "ESC":
                break

            width = prompt_int("Width (px, multiple of 8)", DEFAULT_WIDTH, minimum=64)
            if width is None:
                break
            height = prompt_int("Height (px, multiple of 8)", DEFAULT_HEIGHT, minimum=64)
            if height is None:
                break
            steps = prompt_int("Steps", DEFAULT_STEPS, minimum=1)
            if steps is None:
                break
            cfg_scale = prompt_float("CFG Scale", DEFAULT_CFG_SCALE, minimum=0.0)
            if cfg_scale is None:
                break
            scheduler = select_scheduler_option(scheduler_options, DEFAULT_SCHEDULER)
            if scheduler is None:
                break
            seed_text = get_input(f"{CYAN}Seed (blank=random): {RESET}")
            if seed_text == "ESC":
                break

            seed_val = None
            seed_text = seed_text.strip()
            if seed_text:
                try:
                    seed_val = int(seed_text)
                except ValueError:
                    print(f"{RED}Seed must be an integer{RESET}")
                    refresh_prompt_view = False
                    continue

            width = max(64, (int(width) // 8) * 8)
            height = max(64, (int(height) // 8) * 8)

            try:
                submission = _invoke_generate_image(
                    client,
                    model,
                    prompt,
                    negative,
                    width,
                    height,
                    steps,
                    cfg_scale,
                    scheduler,
                    seed_val,
                    REQUEST_TIMEOUT,
                )
            except InvokeAIClientError as exc:
                print(f"{RED}{exc}{RESET}")
                get_input(f"{CYAN}Press Enter to try again{RESET}")
                refresh_prompt_view = True
                continue
            except requests.RequestException as exc:
                print(f"{RED}Network error: {exc}{RESET}")
                get_input(f"{CYAN}Press Enter to try again{RESET}")
                refresh_prompt_view = True
                continue

            queue_id = submission.get("queue_item_id") if isinstance(submission, dict) else None
            batch_id = submission.get("batch_id") if isinstance(submission, dict) else None
            seed_used = submission.get("seed") if isinstance(submission, dict) else None

            print(f"{GREEN}Prompt sent to {client.nickname}.{RESET}")
            if queue_id:
                print(f"{CYAN}Queue item id: {queue_id}{RESET}")
            else:
                print(
                    f"{YELLOW}Server did not return a queue item id. Check the {TERMINALAI_BOARD_NAME} board for the finished image.{RESET}"
                )
            if batch_id:
                print(f"{CYAN}Batch id: {batch_id}{RESET}")
            if seed_used is not None:
                print(f"{CYAN}Seed used: {seed_used}{RESET}")

            polled_status: Optional[Dict[str, Any]] = None
            if batch_id:
                polled_status = _poll_invoke_batch_status(client, batch_id)
                if isinstance(polled_status, dict):
                    preview = polled_status.get("preview") if isinstance(polled_status.get("preview"), dict) else None
                    preview_error = polled_status.get("preview_error")
                    final_result: Optional[Dict[str, Any]] = None
                    if preview:
                        _present_invoke_preview(preview)
                    else:
                        if _is_batch_complete(polled_status):
                            final_result = _load_invoke_final_result(
                                client,
                                polled_status,
                                batch_id,
                            )
                        if final_result:
                            _present_invoke_result(final_result, label="Batch result available")
                        elif preview_error:
                            print(f"{YELLOW}Preview unavailable: {preview_error}{RESET}")
                        elif _is_batch_complete(polled_status):
                            print(
                                f"{YELLOW}Batch {batch_id} completed without returning a preview. Check the {TERMINALAI_BOARD_NAME} board for the final image.{RESET}"
                            )
                    if not preview and final_result is None and not _is_batch_complete(polled_status):
                        print(
                            f"{YELLOW}Batch {batch_id} is still processing. You can monitor progress on the {TERMINALAI_BOARD_NAME} board.{RESET}"
                        )
                else:
                    print(
                        f"{YELLOW}Unable to retrieve live status for batch {batch_id}. Check the {TERMINALAI_BOARD_NAME} board for updates.{RESET}"
                    )

            acknowledgement = get_input(
                f"{CYAN}Press Enter to prompt again (ESC=change model): {RESET}"
            )
            if isinstance(acknowledgement, str) and acknowledgement.strip().lower() == "esc":
                _forget_model_selection("invokeai")
                break
            refresh_prompt_view = True


def _run_automatic1111_flow(
    client: Automatic1111Client, models: List[Automatic1111Model]
) -> None:
    if not models:
        print(f"{RED}No Automatic1111 models reported by this server{RESET}")
        get_input(f"{CYAN}Press Enter to return{RESET}")
        return

    try:
        sampler_options = client.list_samplers()
    except Automatic1111ClientError as exc:
        print(f"{YELLOW}Failed to retrieve sampler list: {exc}{RESET}")
        sampler_options = []
    except requests.RequestException as exc:
        print(f"{YELLOW}Failed to retrieve sampler list: {exc}{RESET}")
        sampler_options = []

    if not sampler_options:
        sampler_options = [AUTOMATIC1111_DEFAULT_SAMPLER]

    sampler_default = sampler_options[0]

    while True:
        model = select_automatic1111_model(models)
        if model is None:
            return

        try:
            client.set_active_model(model)
        except Automatic1111ClientError as exc:
            print(f"{RED}{exc}{RESET}")
            get_input(f"{CYAN}Press Enter to choose another model{RESET}")
            continue
        except requests.RequestException as exc:
            print(f"{RED}Failed to set active model: {exc}{RESET}")
            get_input(f"{CYAN}Press Enter to choose another model{RESET}")
            continue

        refresh_prompt_view = True
        while True:
            if refresh_prompt_view:
                clear_screen()
                _print_automatic1111_prompt_header(client, model)
            refresh_prompt_view = True

            prompt = get_input(f"{CYAN}Prompt: {RESET}")
            if prompt == "ESC":
                break
            if not prompt.strip():
                print(f"{RED}Prompt cannot be empty{RESET}")
                refresh_prompt_view = False
                continue

            negative = get_input(f"{CYAN}Negative prompt (optional): {RESET}")
            if negative == "ESC":
                break

            width = prompt_int(
                "Width (px, multiple of 8)", AUTOMATIC1111_DEFAULT_WIDTH, minimum=64
            )
            if width is None:
                break
            height = prompt_int(
                "Height (px, multiple of 8)", AUTOMATIC1111_DEFAULT_HEIGHT, minimum=64
            )
            if height is None:
                break
            steps = prompt_int("Steps", AUTOMATIC1111_DEFAULT_STEPS, minimum=1)
            if steps is None:
                break
            cfg_scale = prompt_float(
                "CFG Scale", AUTOMATIC1111_DEFAULT_CFG_SCALE, minimum=0.0
            )
            if cfg_scale is None:
                break
            sampler = select_sampler_option(sampler_options, sampler_default)
            if sampler is None:
                break
            seed_text = get_input(f"{CYAN}Seed (blank=random): {RESET}")
            if seed_text == "ESC":
                break

            seed_val = None
            seed_text = seed_text.strip()
            if seed_text:
                try:
                    seed_val = int(seed_text)
                except ValueError:
                    print(f"{RED}Seed must be an integer{RESET}")
                    refresh_prompt_view = False
                    continue

            try:
                result = _automatic1111_generate_image(
                    client,
                    model,
                    prompt,
                    negative,
                    width,
                    height,
                    steps,
                    cfg_scale,
                    sampler,
                    seed_val,
                    REQUEST_TIMEOUT,
                )
            except Automatic1111ClientError as exc:
                print(f"{RED}{exc}{RESET}")
                get_input(f"{CYAN}Press Enter to try again{RESET}")
                refresh_prompt_view = True
                continue
            except requests.RequestException as exc:
                print(f"{RED}Network error: {exc}{RESET}")
                get_input(f"{CYAN}Press Enter to try again{RESET}")
                refresh_prompt_view = True
                continue

            metadata = result.get("metadata") if isinstance(result, dict) else {}
            seed_used = metadata.get("seed") if isinstance(metadata, dict) else None
            sampler_used = metadata.get("sampler") if isinstance(metadata, dict) else None

            print(f"{GREEN}Image generated on {client.nickname}.{RESET}")
            if seed_used is not None:
                print(f"{CYAN}Seed used: {seed_used}{RESET}")
            if sampler_used:
                print(f"{CYAN}Sampler: {sampler_used}{RESET}")

            _present_invoke_result(result, label="Image generated")

            if sampler_used:
                sampler_default = sampler_used

            acknowledgement = get_input(
                f"{CYAN}Press Enter to prompt again (ESC=change model): {RESET}"
            )
            if isinstance(acknowledgement, str) and acknowledgement.strip().lower() == "esc":
                _forget_model_selection("invokeai")
                break
            refresh_prompt_view = True
def _invoke_generate_image(
    client: InvokeAIClient,
    model: "InvokeAIModel",
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    scheduler: str,
    seed: Optional[int],
    timeout: float,
):
    prompt_text = (prompt or "").strip()
    if not prompt_text:
        raise InvokeAIClientError("Prompt must not be empty")

    board_id = client.ensure_board(TERMINALAI_BOARD_NAME)
    if not _is_valid_terminalai_board_id(board_id):
        raise InvokeAIClientError(TERMINALAI_BOARD_RESOLUTION_ERROR)

    negative_text = (negative_prompt or "").strip()
    scheduler_name = (scheduler or "").strip()
    model_base = getattr(model, "base", "") or ""
    normalized_base = model_base.strip().lower()
    if normalized_base.startswith("flux"):
        if not scheduler_name or scheduler_name.strip().lower() == DEFAULT_SCHEDULER.lower():
            scheduler_name = FLUX_DEFAULT_SCHEDULER
    if not scheduler_name:
        scheduler_name = DEFAULT_SCHEDULER

    return client.submit_image_generation(
        model=model,
        prompt=prompt_text,
        negative_prompt=negative_text,
        width=int(width),
        height=int(height),
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        scheduler=scheduler_name,
        seed=seed,
        board_name=TERMINALAI_BOARD_NAME,
        board_id=board_id,
    )


def _automatic1111_generate_image(
    client: Automatic1111Client,
    model: Automatic1111Model,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    sampler: str,
    seed: Optional[int],
    timeout: float,
):
    prompt_text = (prompt or "").strip()
    if not prompt_text:
        raise Automatic1111ClientError("Prompt must not be empty")

    negative_text = (negative_prompt or "").strip()
    sampler_text = (sampler or "").strip()

    width = max(64, (int(width) // 8) * 8)
    height = max(64, (int(height) // 8) * 8)
    steps = max(1, int(steps))
    cfg_value = float(cfg_scale)

    return client.txt2img(
        model=model,
        prompt=prompt_text,
        negative_prompt=negative_text,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_value,
        sampler=sampler_text or None,
        seed=seed,
        timeout=timeout,
    )


def prompt_int(prompt, default, minimum=None, maximum=None):
    while True:
        resp = get_input(f"{CYAN}{prompt} [{default}]: {RESET}")
        if resp == "ESC":
            return None
        if not resp.strip():
            return default
        try:
            value = int(resp.strip())
        except ValueError:
            print(f"{RED}Please enter a whole number{RESET}")
            continue
        if minimum is not None and value < minimum:
            print(f"{RED}Value must be >= {minimum}{RESET}")
            continue
        if maximum is not None and value > maximum:
            print(f"{RED}Value must be <= {maximum}{RESET}")
            continue
        return value


def prompt_float(prompt, default, minimum=None, maximum=None):
    while True:
        resp = get_input(f"{CYAN}{prompt} [{default}]: {RESET}")
        if resp == "ESC":
            return None
        if not resp.strip():
            return default
        try:
            value = float(resp.strip())
        except ValueError:
            print(f"{RED}Please enter a numeric value{RESET}")
            continue
        if minimum is not None and value < minimum:
            print(f"{RED}Value must be >= {minimum}{RESET}")
            continue
        if maximum is not None and value > maximum:
            print(f"{RED}Value must be <= {maximum}{RESET}")
            continue
        return value


def _prompt_selection(header: str, options: List[str]) -> Optional[int]:
    """Prompt the user to choose from a list of options."""

    if DEBUG_MODE:
        print(f"{CYAN}{header}{RESET}")
        for idx, opt in enumerate(options, 1):
            print(f"{GREEN}{idx}. {opt}{RESET}")
        while True:
            choice = get_input(f"{CYAN}Choose an option: {RESET}")
            if choice == "ESC":
                return None
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(options):
                    return index
            print(f"{RED}Invalid selection{RESET}")
    else:
        sel = interactive_menu(header, options)
        return sel


def choose_mode() -> str:
    options = [
        ("Chat", "chat"),
        ("Imagine", "imagine"),
        ("Configure", "configure"),
        ("[Exit]", "exit"),
    ]
    labels = [label for label, _ in options]
    selection = _prompt_selection("Select Mode:", labels)
    if selection is None:
        return "exit"
    return options[selection][1]


def dispatch_mode(mode_key: str) -> bool:
    """Execute the handler associated with the provided mode key."""

    handler = MODE_DISPATCH.get(mode_key)
    if handler is None:
        print(f"{YELLOW}No handler configured for mode '{mode_key}'.{RESET}")
        return False
    handler()
    return True


def _configure_api_server(api_type: str, label: str) -> Optional[Dict[str, Any]]:
    servers = load_servers(api_type)
    servers = [server for server in servers if api_type in server.get("apis", {})]
    if not servers:
        print(f"{RED}No {label} servers available{RESET}")
        get_input(f"{CYAN}Press Enter to return{RESET}")
        return None

    selected = select_server(servers, allow_back=True)
    if selected is None:
        return None
    _remember_server_selection(api_type, selected)
    print(f"{GREEN}Selected {label} server: {selected['nickname']} ({selected['ip']}){RESET}")
    get_input(f"{CYAN}Press Enter to continue{RESET}")
    return selected


def _configure_ollama_model() -> None:
    server = _configure_api_server("ollama", "Ollama")
    if server is None:
        return

    global SERVER_URL, selected_server, selected_api
    selected_api = "ollama"
    selected_server = server
    SERVER_URL = build_url(server, "ollama")

    models = fetch_models()
    if not models:
        print(f"{YELLOW}No Ollama models found on the selected server{RESET}")
        _forget_model_selection("ollama")
        get_input(f"{CYAN}Press Enter to continue{RESET}")
        return

    preferred = _get_preferred_ollama_model(models)
    if preferred is not None:
        print(f"{GREEN}Current model: {preferred}{RESET}")

    chosen = select_model(models)
    if chosen is None:
        return
    _remember_model_selection("ollama", chosen)
    print(f"{GREEN}Saved Ollama model: {chosen}{RESET}")
    get_input(f"{CYAN}Press Enter to continue{RESET}")


def _configure_invokeai_model() -> None:
    server = _configure_api_server("invokeai", "InvokeAI")
    if server is None:
        return

    selected_port = server.get("apis", {}).get("invokeai")
    if selected_port is None:
        print(f"{RED}Selected server does not expose InvokeAI API{RESET}")
        get_input(f"{CYAN}Press Enter to continue{RESET}")
        return

    client = InvokeAIClient(server["ip"], selected_port, server.get("nickname"), DATA_DIR)
    try:
        models = client.list_models()
    except (InvokeAIClientError, requests.RequestException) as exc:
        print(f"{RED}Failed to retrieve InvokeAI models: {exc}{RESET}")
        _forget_model_selection("invokeai")
        get_input(f"{CYAN}Press Enter to continue{RESET}")
        return

    if not models:
        print(f"{YELLOW}No InvokeAI models found on the selected server{RESET}")
        _forget_model_selection("invokeai")
        get_input(f"{CYAN}Press Enter to continue{RESET}")
        return

    preferred = _get_preferred_invoke_model(models)
    if preferred is not None:
        print(f"{GREEN}Current model: {preferred.name}{RESET}")

    chosen = select_invoke_model(models)
    if chosen is None:
        return
    _remember_model_selection("invokeai", chosen.name, chosen.key)
    print(f"{GREEN}Saved InvokeAI model: {chosen.name}{RESET}")
    get_input(f"{CYAN}Press Enter to continue{RESET}")


def run_configure_menu() -> None:
    options = ["Shodan Scan", "Ollama Server", "Ollama Model", "InvokeAI Server", "InvokeAI Model", "[Back]"]
    while True:
        selection = _prompt_selection("Configure:", options)
        if selection is None or selection == len(options) - 1:
            return
        if selection == 0:
            run_shodan_scan()
            continue
        if selection == 1:
            clear_screen()
            _configure_api_server("ollama", "Ollama")
            clear_screen()
            continue
        if selection == 2:
            clear_screen()
            _configure_ollama_model()
            clear_screen()
            continue
        if selection == 3:
            clear_screen()
            _configure_api_server("invokeai", "InvokeAI")
            clear_screen()
            continue
        if selection == 4:
            clear_screen()
            _configure_invokeai_model()
            clear_screen()
            continue


def run_llm_menu() -> None:
    run_chat_mode()


def run_image_menu() -> None:
    run_image_mode()


def run_automatic1111_mode() -> None:
    global selected_server, selected_api
    selected_api = "automatic1111"

    while True:
        clear_screen()
        servers = load_servers("automatic1111")
        if not servers:
            print(f"{RED}No Automatic1111 servers available{RESET}")
            get_input(f"{CYAN}Press Enter to return to the main menu{RESET}")
            return

        server_choice = select_server(servers, allow_back=True)
        if server_choice is None:
            return
        selected_server = server_choice
        port = selected_server["apis"].get("automatic1111")
        if port is None:
            print(f"{RED}Selected server does not expose Automatic1111 API{RESET}")
            get_input(f"{CYAN}Press Enter to pick another server{RESET}")
            continue

        client = Automatic1111Client(
            selected_server["ip"], port, selected_server["nickname"], DATA_DIR
        )

        try:
            models = client.list_models()
        except Automatic1111ClientError as exc:
            print(f"{RED}{exc}{RESET}")
            try:
                df = read_endpoints(selected_api)
                mask = endpoint_mask(
                    df,
                    selected_server["ip"],
                    selected_server["apis"].get(selected_api),
                )
                if mask.any():
                    df.loc[mask, "is_active"] = False
                    if "inactive_reason" in df.columns:
                        df.loc[mask, "inactive_reason"] = "unsupported models endpoint"
                    write_endpoints(selected_api, df)
                    print(f"{YELLOW}Server marked inactive due to incompatible models API{RESET}")
            except Exception as ex:
                print(f"{RED}Failed to update CSV: {ex}{RESET}")
            get_input(f"{CYAN}Press Enter to pick another server{RESET}")
            continue
        except requests.RequestException as exc:
            print(f"{RED}Failed to retrieve models: {exc}{RESET}")
            try:
                df = read_endpoints(selected_api)
                mask = endpoint_mask(
                    df,
                    selected_server["ip"],
                    selected_server["apis"].get(selected_api),
                )
                if mask.any():
                    df.loc[mask, "is_active"] = False
                    if "inactive_reason" in df.columns:
                        df.loc[mask, "inactive_reason"] = "api unreachable"
                    write_endpoints(selected_api, df)
                    print(f"{YELLOW}Server marked inactive due to network failure{RESET}")
            except Exception as ex:
                print(f"{RED}Failed to update CSV: {ex}{RESET}")
            get_input(f"{CYAN}Press Enter to pick another server{RESET}")
            continue

        if not models:
            print(
                f"{YELLOW}This server did not return any models. Image generation may be unavailable.{RESET}"
            )
            get_input(f"{CYAN}Press Enter to continue{RESET}")
            clear_screen()

        while True:
            options = ["Generate Images", "Rename Server", "[Back]"]
            header = f"Automatic1111 on {client.nickname}:"
            choice = interactive_menu(header, options)
            if choice is None or choice == len(options) - 1:
                break
            if choice == 0:
                clear_screen()
                _run_automatic1111_flow(client, models)
                clear_screen()
                continue
            if choice == 1:
                clear_screen()
                _rename_current_server("automatic1111")
                clear_screen()
                continue


def run_shodan_scan(api_type: Optional[str] = None) -> None:
    clear_screen()
    script_path = Path(__file__).resolve().parent / "shodanscan.py"
    cmd = [sys.executable, str(script_path)]
    if api_type:
        cmd.extend(["--api-type", api_type])
    cmd.extend(sys.argv[1:])
    try:
        subprocess.call(cmd)
    finally:
        if not DEBUG_MODE:
            clear_screen(True)


def load_servers(api_type=None):
    target_types = [api_type] if api_type else list(CSV_PATHS.keys())
    servers = []
    for api in target_types:
        df = read_endpoints(api)
        if df.empty:
            continue
        if "id" in df.columns and "nickname" not in df.columns:
            df = df.rename(columns={"id": "nickname"})
        if "nickname" not in df.columns:
            df["nickname"] = df["ip"]
        if "is_active" not in df.columns:
            df["is_active"] = True
        df["is_active"] = df["is_active"].apply(normalise_bool)
        if "ping" not in df.columns:
            df["ping"] = float("inf")
        df["ping"] = pd.to_numeric(df["ping"], errors="coerce")
        df.loc[df["ping"].isna(), "ping"] = float("inf")
        df = df[df["is_active"]]
        if df.empty:
            continue
        df = df.sort_values("ping")
        grouped = df.groupby("ip", sort=False)
        for ip, group in grouped:
            try:
                ports = {
                    api: int(row["port"])
                    for _, row in group.iterrows()
                    if not pd.isna(row.get("port"))
                }
            except Exception:
                continue
            if not ports:
                continue
            nickname = group.iloc[0]["nickname"]
            country = group.iloc[0].get("country", "")
            ping_val = group.iloc[0].get("ping", float("inf"))
            servers.append(
                {
                    "ip": ip,
                    "nickname": nickname,
                    "apis": ports,
                    "country": country,
                    "ping": ping_val,
                    "api_type": api,
                }
            )
    return servers

def persist_nickname(api_type, server, new_nick):
    port = server["apis"].get(api_type)
    if port is None:
        return
    try:
        df = read_endpoints(api_type)
        if df.empty:
            return
        mask = endpoint_mask(df, server["ip"], port)
        if not mask.any():
            return
        if "nickname" in df.columns:
            df.loc[mask, "nickname"] = new_nick
        if "id" in df.columns:
            df.loc[mask, "id"] = new_nick
        write_endpoints(api_type, df)
    except Exception as e:
        print(f"{RED}Failed to persist nickname: {e}{RESET}")


def _rename_current_server(api_type: str, client: Optional[InvokeAIClient] = None) -> None:
    """Prompt the user to rename the currently selected server."""

    global selected_server

    if not isinstance(selected_server, dict):
        print(f"{YELLOW}Server information unavailable.{RESET}")
        get_input(f"{CYAN}Press Enter to continue{RESET}")
        return

    current_name = selected_server.get("nickname") or selected_server.get("ip") or "Server"
    print(f"{CYAN}Current nickname: {current_name}{RESET}")
    response = get_input(f"{CYAN}New nickname (blank=cancel): {RESET}")
    if isinstance(response, str) and response.strip().lower() == "esc":
        print(f"{YELLOW}Rename cancelled.{RESET}")
        get_input(f"{CYAN}Press Enter to continue{RESET}")
        return

    new_name = response.strip() if isinstance(response, str) else ""
    if not new_name:
        print(f"{YELLOW}Nickname unchanged.{RESET}")
        get_input(f"{CYAN}Press Enter to continue{RESET}")
        return

    selected_server["nickname"] = new_name
    if client is not None:
        client.nickname = new_name
    persist_nickname(api_type, selected_server, new_name)
    print(f"{GREEN}Nickname updated to {new_name}{RESET}")
    get_input(f"{CYAN}Press Enter to continue{RESET}")

def conv_dir(model):
    safe = re.sub(r'[\\/:*?"<>|]', '_', model)
    path = CONV_DIR / safe
    path.mkdir(parents=True, exist_ok=True)
    return path

def list_conversations(model):
    path = conv_dir(model)
    convs = []
    for fn in sorted(os.listdir(path)):
        if fn.endswith(".json"):
            fp = path / fn
            title = fn[:-5]
            try:
                with fp.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    msgs = data.get("messages", [])
                    if msgs:
                        first = msgs[0].get("content", "")
                        first = first.strip()
                        if first:
                            title = (first[:30] + ("..." if len(first) > 30 else ""))
            except Exception:
                pass
            convs.append({"file": fn, "title": title})
    return convs

def load_conversation(model, file):
    path = conv_dir(model) / file
    messages = []
    context = None
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                messages = data.get("messages", [])
                context = data.get("context")
            else:
                messages = data
        except Exception:
            messages = []
            context = None
    history = []
    for i in range(0, len(messages), 2):
        user = messages[i].get("content", "") if i < len(messages) else ""
        ai = messages[i + 1].get("content", "") if i + 1 < len(messages) else ""
        elapsed = messages[i + 1].get("elapsed") if i + 1 < len(messages) else None
        history.append({"user": user, "ai": ai, "elapsed": elapsed})
    return messages, history, context

def save_conversation(model, file, messages, context=None):
    path = conv_dir(model) / file
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump({"messages": messages, "context": context}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"{RED}Failed to save conversation: {e}{RESET}")

def create_conversation_file(model, first_prompt):
    safe_prompt = re.sub(r'[\\/:*?"<>|]', '_', first_prompt.strip())
    safe_prompt = safe_prompt[:30]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{safe_prompt}.json"
    # ensure directory exists
    conv_dir(model)
    return filename

def has_conversations(model):
    path = conv_dir(model)
    return any(fn.endswith(".json") for fn in os.listdir(path))


def ping_time(ip, port):
    """Measure latency to an IP and port using a TCP handshake."""
    try:
        start = time.time()
        with socket.create_connection((ip, int(port)), timeout=1):
            end = time.time()
        return (end - start) * 1000
    except Exception:
        return None


def check_ollama_api(ip, port):
    """Verify the Ollama API responds on the host and port."""
    try:
        r = requests.get(f"http://{ip}:{port}/api/tags", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def check_invoke_api(ip, port):
    """Verify an InvokeAI server responds at the provided host and port."""
    client = InvokeAIClient(ip, int(port), data_dir=DATA_DIR)
    try:
        client.check_health()
        return True
    except InvokeAIClientError:
        return False
    except requests.RequestException:
        return False


def check_automatic1111_api(ip, port):
    """Verify an Automatic1111 server responds at the provided host and port."""

    client = Automatic1111Client(ip, int(port), data_dir=DATA_DIR)
    try:
        return client.check_health()
    except Automatic1111ClientError:
        return False
    except requests.RequestException:
        return False


def _probe_endpoint(api_type, ip, port, perform_api_check):
    latency = ping_time(ip, port)
    api_ok = None
    if latency is not None and perform_api_check:
        if api_type == "invokeai":
            api_ok = check_invoke_api(ip, port)
        elif api_type == "automatic1111":
            api_ok = check_automatic1111_api(ip, port)
        else:
            api_ok = check_ollama_api(ip, port)
    return latency, api_ok


def update_pings(
    target_api=None,
    *,
    verify_api=True,
    update_activity=True,
    max_workers=8,
):
    apis = [target_api] if target_api else list(CSV_PATHS.keys())
    for api in apis:
        df = read_endpoints(api)
        if df.empty:
            continue
        if "ping" not in df.columns:
            df["ping"] = pd.NA
        if "is_active" not in df.columns:
            df["is_active"] = True
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        candidates = []
        for idx, row in df.iterrows():
            currently_active = normalise_bool(row.get("is_active", True))
            if update_activity and not currently_active:
                continue

            ip = str(row.get("ip", "")).strip()
            port_val = row.get("port")
            try:
                port = int(port_val)
            except (TypeError, ValueError):
                port = None

            if not ip or port is None:
                df.at[idx, "ping"] = pd.NA
                if update_activity and currently_active:
                    df.at[idx, "is_active"] = False
                    if "inactive_reason" in df.columns:
                        df.at[idx, "inactive_reason"] = "invalid endpoint"
                if "last_check_date" in df.columns:
                    df.at[idx, "last_check_date"] = now
                continue

            perform_api_check = verify_api and update_activity
            candidates.append((idx, ip, port, currently_active, perform_api_check))

        if not candidates:
            write_endpoints(api, df)
            continue

        worker_count = max(1, min(max_workers, len(candidates)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {}
            for idx, ip, port, currently_active, perform_api_check in candidates:
                future = executor.submit(
                    _probe_endpoint,
                    api,
                    ip,
                    port,
                    perform_api_check,
                )
                future_map[future] = (idx, currently_active)

            for future in as_completed(future_map):
                idx, currently_active = future_map[future]
                latency, api_ok = future.result()

                if latency is None:
                    df.at[idx, "ping"] = pd.NA
                    if update_activity and currently_active:
                        df.at[idx, "is_active"] = False
                        if "inactive_reason" in df.columns:
                            df.at[idx, "inactive_reason"] = "ping timeout"
                else:
                    df.at[idx, "ping"] = latency
                    if update_activity and api_ok is not None:
                        df.at[idx, "is_active"] = api_ok
                        if "inactive_reason" in df.columns:
                            df.at[idx, "inactive_reason"] = (
                                "" if api_ok else "api unreachable"
                            )

                if "last_check_date" in df.columns:
                    df.at[idx, "last_check_date"] = now

        write_endpoints(api, df)

def select_server(servers, allow_back=False):
    if DEBUG_MODE:
        print(f"{CYAN}Available Servers:{RESET}")
        for i, s in enumerate(servers, 1):
            ping_val = s.get("ping", float("inf"))
            ping_str = "?" if ping_val == float("inf") else f"{ping_val:.1f} ms"
            color = WARNING if ping_str == "?" else GREEN
            print(f"{color}{i}. {s['nickname']} ({s['ip']}) - {ping_str}{RESET}")
        if allow_back:
            print(f"{GREEN}0. [Back]{RESET}")
        while True:
            c = get_input(f"{CYAN}Select server: {RESET}")
            if c == "ESC":
                if allow_back:
                    return None
                confirm = get_input(f"{CYAN}Exit? y/n: {RESET}")
                if confirm.lower() == "y":
                    sys.exit(0)
                else:
                    continue
            if allow_back and c == "0":
                return None
            if c.isdigit() and 1 <= int(c) <= len(servers):
                return servers[int(c) - 1]
            print(f"{RED}Invalid selection{RESET}")
    else:
        options = []
        for s in servers:
            ping_val = s.get("ping", float("inf"))
            ping_str = "?" if ping_val == float("inf") else f"{ping_val:.1f} ms"
            style = "warning" if ping_str == "?" else "default"
            options.append({"label": f"{s['nickname']} ({s['ip']}) - {ping_str}", "style": style})
        if allow_back:
            options.append({"label": "[Back]"})
        while True:
            choice = interactive_menu("Available Servers:", options)
            if choice is None:
                if allow_back:
                    return None
                if confirm_exit():
                    sys.exit(0)
                else:
                    continue
            if allow_back and choice == len(options) - 1:
                return None
            return servers[choice]

def select_model(models):
    if DEBUG_MODE:
        print(f"{CYAN}Available Models:{RESET}")
        for i, model in enumerate(models, 1):
            mark = " *" if has_conversations(model) else ""
            embed_label = " [embed]" if is_embedding_model(model) else ""
            print(f"{GREEN}{i}. {model}{embed_label}{mark}{RESET}")
        print(f"{GREEN}0. [Back]{RESET}")
        while True:
            c = get_input(f"{CYAN}Select model: {RESET}")
            if c in ("ESC", "0") or c.lower() == "back":
                return None
            if c.isdigit() and 1 <= int(c) <= len(models):
                return models[int(c) - 1]
            print(f"{RED}Invalid selection{RESET}")
    else:
        options = []
        for model in models:
            mark = " *" if has_conversations(model) else ""
            embed_label = " [embed]" if is_embedding_model(model) else ""
            options.append(f"{model}{embed_label}{mark}")
        while True:
            choice = interactive_menu("Available Models:", options)
            if choice is None:
                return None
            return models[choice]


def select_invoke_model(models):
    labels = [
        f"{m.name} [{m.base or 'unknown'}]" if isinstance(m.base, str) else m.name
        for m in models
    ]
    if DEBUG_MODE:
        print(f"{CYAN}Available InvokeAI Models:{RESET}")
        for idx, label in enumerate(labels, 1):
            print(f"{GREEN}{idx}. {label}{RESET}")
        print(f"{GREEN}0. [Back]{RESET}")
        while True:
            choice = get_input(f"{CYAN}Select model: {RESET}")
            if choice in ("ESC", "0") or choice.lower() == "back":
                return None
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                return models[int(choice) - 1]
            print(f"{RED}Invalid selection{RESET}")
    else:
        while True:
            choice = interactive_menu("InvokeAI Models:", labels)
            if choice is None:
                return None
            return models[choice]


def select_automatic1111_model(models: List[Automatic1111Model]) -> Optional[Automatic1111Model]:
    labels: List[str] = []
    for model in models:
        pieces: List[str] = []
        if isinstance(model.title, str) and model.title.strip():
            pieces.append(model.title.strip())
        if isinstance(model.name, str):
            name_text = model.name.strip()
            if name_text and name_text not in pieces:
                pieces.append(name_text)
        labels.append(" • ".join(pieces) if pieces else model.name or "Unnamed model")

    if DEBUG_MODE:
        print(f"{CYAN}Available Automatic1111 Models:{RESET}")
        for idx, label in enumerate(labels, 1):
            print(f"{GREEN}{idx}. {label}{RESET}")
        print(f"{GREEN}0. [Back]{RESET}")
        while True:
            choice = get_input(f"{CYAN}Select model: {RESET}")
            if choice in ("ESC", "0") or choice.lower() == "back":
                return None
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                return models[int(choice) - 1]
            print(f"{RED}Invalid selection{RESET}")
    else:
        while True:
            choice = interactive_menu("Automatic1111 Models:", labels)
            if choice is None:
                return None
            return models[choice]


def _select_freeform_option(
    label: str, options: List[str], default_value: str
) -> Optional[str]:
    """Prompt for a free-form text option with optional suggestions."""

    cleaned: List[str] = []
    seen: set[str] = set()
    for opt in options:
        if not isinstance(opt, str):
            continue
        text = opt.strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        cleaned.append(text)

    if cleaned and DEBUG_MODE:
        print(f"{CYAN}Available {label.lower()} options:{RESET}")
        for idx, value in enumerate(cleaned, 1):
            print(f"{GREEN}{idx}. {value}{RESET}")

    while True:
        entry = get_input(f"{CYAN}{label} [{default_value}]: {RESET}")
        if entry == "ESC":
            return None
        entry = entry.strip()
        if not entry:
            return default_value
        return entry


def select_scheduler_option(options: List[str], default_value: str) -> Optional[str]:
    """Prompt for a scheduler using free text input."""

    return _select_freeform_option("Scheduler", options, default_value)


def select_sampler_option(options: List[str], default_value: str) -> Optional[str]:
    """Prompt for a sampler using free text input."""

    return _select_freeform_option("Sampler", options, default_value)


def fetch_models():
    try:
        r = requests.get(f"{SERVER_URL}/v1/models", timeout=5)
        r.raise_for_status()
        data = r.json()
        models = []
        for m in data.get("models", []) or data.get("data", []) or []:
            if isinstance(m, dict):
                models.append(m.get("id") or m.get("name") or "")
            elif isinstance(m, str):
                models.append(m)
        models = [m for m in models if m]
        seen_changes = False
        for model_name in models:
            _, created = _ensure_model_entry(model_name)
            if created:
                seen_changes = True
        if seen_changes:
            _save_model_capabilities()
        try:
            tags_resp = requests.get(f"{SERVER_URL}/api/tags", timeout=5)
            tags_resp.raise_for_status()
            tags_payload = tags_resp.json()
            tag_models = tags_payload.get("models")
            if isinstance(tag_models, list):
                for entry in tag_models:
                    if isinstance(entry, dict):
                        name = entry.get("name") or entry.get("model") or entry.get("id")
                        if name:
                            update_model_capabilities_from_metadata(name, entry)
        except (requests.RequestException, ValueError):
            pass
        return models
    except Exception as e:
        print(f"{RED}Failed to fetch models: {e}{RESET}")
        return []



def try_embeddings_request(model, prompt_text, headers, timeout_val):
    payload = {"model": model, "prompt": prompt_text, "input": prompt_text}
    try:
        response = requests.post(
            f"{SERVER_URL}/api/embeddings",
            headers=headers,
            json=payload,
            timeout=(timeout_val, timeout_val),
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print(f"\n{RED}/api/embeddings request timed out after {timeout_val}s{RESET}")
        return None, True, True
    except requests.exceptions.HTTPError as error:
        message = describe_http_error(error)
        print(f"\n{RED}Failed /api/embeddings: {message}{RESET}")
        status = error.response.status_code if error.response is not None else None
        failure = bool(status and status >= 500)
        return None, failure, False
    except requests.exceptions.RequestException as error:
        print(f"\n{RED}Failed /api/embeddings: {error}{RESET}")
        return None, True, False

    try:
        data = response.json()
    except ValueError:
        print(f"\n{RED}Failed /api/embeddings: invalid JSON response{RESET}")
        return None, False, False

    vector = extract_embedding_vector(data)
    if vector is None:
        print(f"\n{RED}/api/embeddings response did not include embedding values{RESET}")
        return None, False, False

    mark_model_as_embedding(model)
    return format_embedding_response(vector), False, False


def build_url(server, api):
    return f"http://{server['ip']}:{server['apis'][api]}"

def redraw_ui(model):
    clear_screen()
    ip = selected_server['ip']
    port = selected_server['apis'][selected_api]
    print(f"{BOLD}{GREEN}🖥️ AI Terminal Interface | Active Model: {model}{RESET}")
    print(f"{GREEN}{ip}:{port} | {selected_server['nickname']}{RESET}")
    print(f"{YELLOW}Type prompts below. Commands: /exit, /clear, /paste, /back, /print, /nick (Esc=Back){RESET}")


def _print_invoke_prompt_header(client: InvokeAIClient, model: InvokeAIModel) -> None:
    """Render the InvokeAI prompt header bar."""

    nickname = getattr(client, "nickname", None) or client.ip
    endpoint = f"{client.ip}:{client.port}" if getattr(client, "port", None) else client.ip
    print(f"{BOLD}{GREEN}🖼️ InvokeAI Image Generator{RESET}")
    print(f"{GREEN}Server: {nickname} ({endpoint}){RESET}")
    print(f"{GREEN}Model: {model.name}{RESET}")
    print(
        f"{YELLOW}Enter prompt details below. Press Esc at any prompt to change the model or server.{RESET}"
    )
    print()


def _print_automatic1111_prompt_header(
    client: Automatic1111Client, model: Automatic1111Model
) -> None:
    """Render the Automatic1111 prompt header bar."""

    nickname = getattr(client, "nickname", None) or client.ip
    endpoint = f"{client.ip}:{client.port}" if getattr(client, "port", None) else client.ip
    print(f"{BOLD}{GREEN}🖼️ Automatic1111 Image Generator{RESET}")
    print(f"{GREEN}Server: {nickname} ({endpoint}){RESET}")
    title = model.title if isinstance(model.title, str) and model.title.strip() else model.name
    print(f"{GREEN}Model: {title}{RESET}")
    print(
        f"{YELLOW}Enter prompt details below. Press Esc at any prompt to change the model or server.{RESET}"
    )
    print()


def _print_board_view_header(client: InvokeAIClient, board_name: str) -> None:
    """Render the header for the board image viewer."""

    nickname = getattr(client, "nickname", None) or client.ip
    endpoint = f"{client.ip}:{client.port}" if getattr(client, "port", None) else client.ip
    print(f"{BOLD}{GREEN}🖼️ InvokeAI Board Viewer{RESET}")
    print(f"{GREEN}Server: {nickname} ({endpoint}){RESET}")
    print(f"{GREEN}Board: {board_name}{RESET}")
    print()

def display_connecting_box(x, y, w, h):
    for i in range(h):
        print(f"\033[{y+i};{x}H{' '*w}{RESET}")
    sys.stdout.flush()
    print(f"\033[{y};{x}H{GREEN}┌{'─'*(w-2)}┐{RESET}")
    for i in range(1, h-1):
        print(f"\033[{y+i};{x}H{GREEN}│{' '*(w-2)}│{RESET}")
    print(f"\033[{y+h-1};{x}H{GREEN}└{'─'*(w-2)}┘{RESET}")
    msg = "HACK THE PLANET"
    mx = x + (w - len(msg)) // 2
    my = y + h // 2
    print(f"\033[{my};{mx}H{BOLD}{GREEN}{msg}{RESET}", flush=True)
    time.sleep(CONNECTING_SCREEN_DURATION)


def render_markdown(text):
    lines = text.splitlines()
    in_think = False
    in_code = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("<think>"):
            # Skip over any internal reasoning blocks without
            # emitting an extra "Thinking..." line. The thinking
            # timer already communicates that status to the user,
            # and showing it here causes duplicate output.
            in_think = True
            continue
        if s.endswith("</think>"):
            in_think = False
            continue
        if in_think:
            continue
        if s == "</s>":
            print("-" * 60)
            continue
        if s.startswith("```"):
            in_code = not in_code
            print("=" * 60)
            continue
        out = ln.replace("</s>", "")
        if not in_code:
            out = out.lstrip()
        print(out)
    print()

def reprint_history(history):
    for e in history:
        print(f"🧑 : {e['user']}")
        wait = e.get("elapsed")
        if wait is not None:
            print(f"{AI_COLOR}🖥️ : Thinking... ({wait}s){RESET}")
        else:
            print(f"{AI_COLOR}🖥️ : {RESET}")
        render_markdown(e['ai'])

def select_conversation(model):
    convs = list_conversations(model)
    if not convs:
        print(f"{CYAN}No previous conversations found.{RESET}")
        return None, [], [], None

    if DEBUG_MODE:
        print(f"{CYAN}Conversations:{RESET}")
        print(f"{GREEN}1. Start new conversation{RESET}")
        for i, c in enumerate(convs, 2):
            print(f"{GREEN}{i}. {c['title']}{RESET}")
        print(f"{GREEN}0. [Back]{RESET}")

        while True:
            choice = get_input(f"{CYAN}Select conversation: {RESET}")
            if choice in ("ESC", "0") or choice.lower() == "back":
                return 'back', None, None, None
            if choice == '1':
                return None, [], [], None
            if choice.isdigit() and 2 <= int(choice) < len(convs) + 2:
                file = convs[int(choice) - 2]['file']
                messages, history, context = load_conversation(model, file)
                return file, messages, history, context
            print(f"{RED}Invalid selection{RESET}")
    else:
        options = ["Start new conversation"] + [c['title'] for c in convs]
        while True:
            choice = interactive_menu("Conversations:", options)
            if choice is None:
                return 'back', None, None, None
            if choice == 0:
                return None, [], [], None
            file = convs[choice - 1]['file']
            messages, history, context = load_conversation(model, file)
            return file, messages, history, context

def chat_loop(model, conv_file, messages=None, history=None, context=None):
    global SERVER_URL, selected_server, selected_api
    redraw_ui(model)
    if messages is None:
        messages = []
    if history is None:
        history = []
    if history:
        reprint_history(history)
    try:
        while True:
            start = time.time()
            sys.stdout.write(f"{RESET}\U0001f9d1 : ")
            sys.stdout.flush()
            user_input = ""
            if os.name == "nt":
                while True:
                    if time.time() - start > IDLE_TIMEOUT:
                        if not DEBUG_MODE:
                            rain(persistent=True, use_alt_screen=True)
                        start = time.time()
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch in ('\r', '\n'):
                            print()
                            break
                        elif ch == '\x1b':
                            print()
                            return "back"
                        elif ch == '\b':
                            if user_input:
                                user_input = user_input[:-1]
                                sys.stdout.write('\b \b')
                                sys.stdout.flush()
                        else:
                            user_input += ch
                            sys.stdout.write(ch)
                            sys.stdout.flush()
                        start = time.time()
                    else:
                        time.sleep(0.1)
                user_input = user_input.strip()
                if not user_input:
                    continue
            else:
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                try:
                    while True:
                        if time.time() - start > IDLE_TIMEOUT:
                            if not DEBUG_MODE:
                                rain(persistent=True, use_alt_screen=True)
                                tty.setcbreak(fd)
                            start = time.time()
                        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if rlist:
                            ch = sys.stdin.read(1)
                            if ch in ('\n', '\r'):
                                print()
                                break
                            elif ch == '\x7f':
                                if user_input:
                                    user_input = user_input[:-1]
                                    sys.stdout.write('\b \b')
                                    sys.stdout.flush()
                            elif ch == '\x1b':
                                print()
                                return "back"
                            else:
                                user_input += ch
                                sys.stdout.write(ch)
                                sys.stdout.flush()
                            start = time.time()
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                user_input = user_input.strip()
                if not user_input:
                    continue

            cmd = user_input.lower()
            if cmd == "/exit":
                return "exit"
            elif cmd == "/clear":
                clear_screen(force=True)
                redraw_ui(model)
                continue
            elif cmd == "/print":
                if history:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    LOG_DIR.mkdir(parents=True, exist_ok=True)
                    fn = LOG_DIR / f"chat_{ts}.txt"
                    with fn.open("w", encoding="utf-8") as f:
                        for h in history:
                            elapsed = h.get("elapsed")
                            if elapsed is None:
                                f.write(
                                    f"User: {h['user']}\nAI:\n{h['ai']}\n\n"
                                )
                            else:
                                f.write(
                                    f"User: {h['user']}\nAI: Thinking... ({elapsed}s)\n{h['ai']}\n\n"
                                )
                    print(f"{YELLOW}Saved to {fn}{RESET}")
                else:
                    print(f"{RED}No history{RESET}")
                continue
            elif cmd == "/paste":
                print(f"{CYAN}Paste (Ctrl+D/Ctrl+Z to end){RESET}")
                lines = []
                try:
                    while True:
                        lines.append(input())
                except:
                    pass
                user_input = "".join(lines).strip()

            elif cmd == "/back":
                return "back"

            elif cmd == "/nick":
                new = input(f"{CYAN}New nickname: {RESET}").strip()
                if new:
                    selected_server["nickname"] = new
                    persist_nickname(selected_api, selected_server, new)
                    print(f"{YELLOW}Nickname saved{RESET}")
                redraw_ui(model)
                continue
            start, stop_event = start_thinking_timer()
            headers = api_headers()
            resp = None
            server_failed = False
            timed_out = False
            embedding_only = False
            timeout_val = REQUEST_TIMEOUT * (len(messages) + 1)

            chat_paths = [
                "/v1/chat/completions",
                "/v1/chat",
                "/api/chat",
                "/chat",
            ]

            embedding_candidate = is_embedding_model(model)
            embedding_info = MODEL_CAPABILITIES.get(model, {})
            embedding_confirmed = bool(embedding_info.get("confirmed"))

            if not embedding_confirmed:
                for p in chat_paths:
                    try:
                        payload = {
                            "model": model,
                            "messages": messages + [{"role": "user", "content": user_input}],
                            "stream": False,
                        }
                        r = requests.post(
                            f"{SERVER_URL}{p}",
                            headers=headers,
                            json=payload,
                            timeout=(timeout_val, timeout_val),
                        )
                        r.raise_for_status()
                        data = r.json()
                        msg = (
                            data.get("choices", [{}])[0].get("message", {}).get("content")
                            or data.get("choices", [{}])[0].get("text")
                            or data.get("completion")
                        )
                        if msg:
                            resp = msg
                            break
                    except requests.exceptions.Timeout:
                        print(f"\n{RED}{p} request timed out after {timeout_val}s{RESET}")
                        server_failed = True
                        timed_out = True
                        break
                    except requests.exceptions.HTTPError as e:
                        message = describe_http_error(e)
                        print(f"\n{RED}Failed {p}: {message}{RESET}")
                        status = e.response.status_code if e.response is not None else None
                        if status in (400, 404):
                            embedding_candidate = True
                        continue
                    except requests.exceptions.RequestException as e:
                        print(f"\n{RED}Failed {p}: {e}{RESET}")
                        server_failed = True
                        break

            if not resp and not embedding_confirmed:
                gen = [
                    "/v1/completions",
                    "/v1/complete",
                    "/api/generate",
                    "/generate",
                ]
                for p in gen:
                    try:
                        if p == "/api/generate":
                            payload = {"model": model, "prompt": user_input, "stream": False}
                            if context:
                                payload["context"] = context
                        else:
                            prompt_text = "\n".join([
                                f"{m['role']}: {m['content']}" for m in messages + [{"role": "user", "content": user_input}]
                            ])
                            payload = {"model": model, "prompt": prompt_text, "stream": False}
                        r = requests.post(
                            f"{SERVER_URL}{p}",
                            headers=headers,
                            json=payload,
                            timeout=(timeout_val, timeout_val),
                        )
                        r.raise_for_status()
                        data = r.json()
                        msg = (
                            data.get("choices", [{}])[0].get("text")
                            or data.get("completion")
                            or data.get("response")
                        )
                        if msg:
                            resp = msg
                            if "context" in data:
                                context = data.get("context")
                            break
                    except requests.exceptions.Timeout:
                        print(f"\n{RED}{p} request timed out after {timeout_val}s{RESET}")
                        server_failed = True
                        timed_out = True
                        break
                    except requests.exceptions.HTTPError as e:
                        message = describe_http_error(e)
                        print(f"\n{RED}Failed {p}: {message}{RESET}")
                        status = e.response.status_code if e.response is not None else None
                        if status in (400, 404):
                            embedding_candidate = True
                        continue
                    except requests.exceptions.RequestException as e:
                        print(f"\n{RED}Failed {p}: {e}{RESET}")
                        server_failed = True
                        break

            if not resp:
                if embedding_candidate:
                    embedding_only = True
                    mark_model_as_embedding(model)
                else:
                    print(f"{RED}[Error] No response{RESET}")
                    if not server_failed:
                        server_failed = True

            elapsed = stop_thinking_timer(start, stop_event, timed_out)

            if embedding_only:
                print(
                    f"{YELLOW}Model '{model}' appears to be embedding-only and cannot be used for chat.{RESET}"
                )
                get_input(f"{CYAN}Press Enter to choose another model{RESET}")
                return "embedding_only"

            if server_failed:
                try:
                    df = read_endpoints(selected_api)
                    mask = endpoint_mask(
                        df,
                        selected_server["ip"],
                        selected_server["apis"][selected_api],
                    )
                    if mask.any():
                        df.loc[mask, "is_active"] = False
                        if "inactive_reason" in df.columns:
                            df.loc[mask, "inactive_reason"] = "api unreachable"
                        write_endpoints(selected_api, df)
                    print(f"{RED}Server marked inactive due to failure{RESET}")
                except Exception as ex:
                    print(f"{RED}Failed to update CSV: {ex}{RESET}")
                return "server_inactive"

            if not timed_out and resp:
                # Response prints without an extra AI prefix so that
                # history and live conversation share the same
                # formatting: a thinking line followed by the reply.
                render_markdown(resp)
                if conv_file is None:
                    conv_file = create_conversation_file(model, user_input)
                messages.append({"role": "user", "content": user_input})
                messages.append(
                    {"role": "assistant", "content": resp, "elapsed": elapsed}
                )
                history.append({"user": user_input, "ai": resp, "elapsed": elapsed})
                save_conversation(model, conv_file, messages, context)

    except KeyboardInterrupt:
        print(f"{YELLOW}Session interrupted{RESET}")
        if history:
            save = input(f"{CYAN}Save log? (y/n): {RESET}").lower()
            if save == "y":
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                fn = LOG_DIR / f"chat_{ts}.txt"
                with fn.open("w", encoding="utf-8") as f:
                    for h in history:
                        elapsed = h.get("elapsed")
                        if elapsed is None:
                            f.write(
                                f"User: {h['user']}\nAI:\n{h['ai']}\n\n"
                            )
                        else:
                            f.write(
                                f"User: {h['user']}\nAI: Thinking... ({elapsed}s)\n{h['ai']}\n\n"
                            )
                print(f"Saved to {fn}")

def run_chat_mode():
    global SERVER_URL, selected_server, selected_api
    selected_api = "ollama"
    while True:
        clear_screen()
        choice = _choose_server_for_api("ollama", allow_back=True)
        if choice is None:
            print(f"{RED}No active Ollama servers available{RESET}")
            get_input(f"{CYAN}Press Enter to return to the main menu{RESET}")
            return
        selected_server = choice
        clear_screen()
        SERVER_URL = build_url(selected_server, selected_api)

        models = fetch_models()
        if not models:
            print(f"{RED}No models found on selected server{RESET}")
            _forget_server_selection("ollama")
            try:
                df = read_endpoints(selected_api)
                mask = endpoint_mask(
                    df,
                    selected_server["ip"],
                    selected_server["apis"][selected_api],
                )
                if mask.any():
                    df.loc[mask, "is_active"] = False
                    if "inactive_reason" in df.columns:
                        df.loc[mask, "inactive_reason"] = "no models"
                    write_endpoints(selected_api, df)
            except Exception as ex:
                print(f"{RED}Failed to update CSV: {ex}{RESET}")
            get_input(f"{CYAN}Press Enter to select another server{RESET}")
            continue

        while True:
            chosen = _get_preferred_ollama_model(models)
            if chosen is None:
                chosen = select_model(models)
                if chosen is None:
                    clear_screen()
                    return
            _remember_model_selection("ollama", chosen)

            embedding_info = MODEL_CAPABILITIES.get(chosen, {})
            embedding_confirmed = bool(embedding_info.get("confirmed"))
            embedding_detected = embedding_confirmed or is_embedding_model(chosen)
            if embedding_detected:
                clear_screen()
                mark_model_as_embedding(chosen)
                _forget_model_selection("ollama")
                print(
                    f"{YELLOW}Model '{chosen}' is an embedding model and cannot be used for chat.{RESET}"
                )
                get_input(f"{CYAN}Press Enter to choose another model{RESET}")
                clear_screen()
                continue
            clear_screen()
            while True:
                if has_conversations(chosen):
                    conv_file, messages, history, context = select_conversation(chosen)
                    if conv_file == "back":
                        _forget_model_selection("ollama")
                        break
                else:
                    print(f"{CYAN}No previous conversations found.{RESET}")
                    conv_file, messages, history, context = (None, [], [], None)

                result = chat_loop(chosen, conv_file, messages, history, context)
                if result == "back":
                    clear_screen()
                    if has_conversations(chosen):
                        continue
                    conv_file = "back"
                    _forget_model_selection("ollama")
                    break
                if result == "server_inactive":
                    conv_file = "server_inactive"
                    break
                if result == "embedding_only":
                    conv_file = "embedding_only"
                    _forget_model_selection("ollama")
                    break
                if result == "exit":
                    sys.exit(0)
                else:
                    return
            if conv_file == "back":
                clear_screen()
                continue
            if conv_file == "server_inactive":
                clear_screen()
                break
            if conv_file == "embedding_only":
                clear_screen()
                continue
            break


def run_image_mode():
    global SERVER_URL, selected_server, selected_api
    selected_api = "invokeai"
    while True:
        clear_screen()
        server_choice = _choose_server_for_api("invokeai", allow_back=True)
        if server_choice is None:
            print(f"{RED}No InvokeAI servers available{RESET}")
            get_input(f"{CYAN}Press Enter to return to the main menu{RESET}")
            return
        selected_server = server_choice
        port = selected_server["apis"].get("invokeai")
        if port is None:
            print(f"{RED}Selected server does not expose InvokeAI API{RESET}")
            get_input(f"{CYAN}Press Enter to pick another server{RESET}")
            continue

        client = InvokeAIClient(selected_server["ip"], port, selected_server["nickname"], DATA_DIR)
        try:
            client.check_health()
        except InvokeAIClientError as exc:
            print(f"{RED}{exc}{RESET}")
            _forget_server_selection("invokeai")
            get_input(f"{CYAN}Press Enter to pick another server{RESET}")
            continue
        except requests.RequestException as exc:
            print(f"{RED}Network error while verifying InvokeAI server: {exc}{RESET}")
            _forget_server_selection("invokeai")
            get_input(f"{CYAN}Press Enter to pick another server{RESET}")
            continue

        try:
            board_id = client.ensure_board(TERMINALAI_BOARD_NAME)
        except InvokeAIClientError as exc:
            print(f"{RED}{exc}{RESET}")
            _forget_server_selection("invokeai")
            get_input(f"{CYAN}Press Enter to pick another server{RESET}")
            continue
        if not _is_valid_terminalai_board_id(board_id):
            print(f"{RED}{TERMINALAI_BOARD_ID_ERROR_MESSAGE}{RESET}")
            _forget_server_selection("invokeai")
            get_input(f"{CYAN}Press Enter to pick another server{RESET}")
            continue

        normalized_board_id = _normalize_board_id(board_id) or board_id
        print(
            f"{GREEN}Images will be saved to board {TERMINALAI_BOARD_NAME} (id: {normalized_board_id}).{RESET}"
        )
        try:
            models = client.list_models()
        except InvokeAIClientError as exc:
            print(f"{RED}{exc}{RESET}")
            _forget_server_selection("invokeai")
            try:
                df = read_endpoints(selected_api)
                mask = endpoint_mask(
                    df,
                    selected_server["ip"],
                    selected_server["apis"][selected_api],
                )
                if mask.any():
                    df.loc[mask, "is_active"] = False
                    if "inactive_reason" in df.columns:
                        df.loc[mask, "inactive_reason"] = "unsupported models endpoint"
                    write_endpoints(selected_api, df)
                    print(f"{YELLOW}Server marked inactive due to incompatible models API{RESET}")
            except Exception as ex:
                print(f"{RED}Failed to update CSV: {ex}{RESET}")
            get_input(f"{CYAN}Press Enter to pick another server{RESET}")
            continue
        except requests.RequestException as exc:
            print(f"{RED}Failed to retrieve models: {exc}{RESET}")
            _forget_server_selection("invokeai")
            try:
                df = read_endpoints(selected_api)
                mask = endpoint_mask(
                    df,
                    selected_server["ip"],
                    selected_server["apis"][selected_api],
                )
                if mask.any():
                    df.loc[mask, "is_active"] = False
                    if "inactive_reason" in df.columns:
                        df.loc[mask, "inactive_reason"] = "api unreachable"
                    write_endpoints(selected_api, df)
                    print(f"{YELLOW}Server marked inactive due to network failure{RESET}")
            except Exception as ex:
                print(f"{RED}Failed to update CSV: {ex}{RESET}")
            get_input(f"{CYAN}Press Enter to pick another server{RESET}")
            continue
        if not models:
            print(f"{YELLOW}This server did not return any models. Image generation may be unavailable.{RESET}")
            get_input(f"{CYAN}Press Enter to continue{RESET}")
            clear_screen()

        while True:
            options = ["View Boards", "Generate Images", "Rename Server", "[Back]"]
            header = f"InvokeAI on {client.nickname}:"
            choice = interactive_menu(header, options)
            if choice is None or choice == len(options) - 1:
                return
            if choice == 0:
                clear_screen()
                _view_server_boards(client)
                clear_screen()
                continue
            if choice == 1:
                clear_screen()
                _run_generation_flow(client, models)
                clear_screen()
                continue
            if choice == 2:
                clear_screen()
                _rename_current_server("invokeai", client)
                clear_screen()
                continue


# Main loop
MODE_DISPATCH.update(
    {
        "chat": run_chat_mode,
        "imagine": run_image_mode,
        "configure": run_configure_menu,
        "llm": run_llm_menu,
        "llm-ollama": run_chat_mode,
        "image": run_image_menu,
        "image-invokeai": run_image_mode,
        "image-automatic1111": run_automatic1111_mode,
        "shodan": run_shodan_scan,
    }
)

if __name__ == "__main__":
    selected_api = "ollama"

    if not DEBUG_MODE:
        clear_screen()
        cols, rows = shutil.get_terminal_size(fallback=(80, 24))
        box_w, box_h = 30, 5
        box_x = (cols - box_w) // 2 + 1
        box_y = (rows - box_h) // 2 + 1
        stop_rain = threading.Event()
        rain_thread = threading.Thread(
            target=rain,
            kwargs={
                "persistent": True,
                "stop_event": stop_rain,
                "box_top": box_y,
                "box_bottom": box_y + box_h - 1,
                "box_left": box_x,
                "box_right": box_x + box_w - 1,
                "clear_screen": False,
            },
        )
        rain_thread.start()
        display_connecting_box(box_x, box_y, box_w, box_h)
        ping_thread = threading.Thread(
            target=update_pings,
            kwargs={
                "verify_api": False,
                "update_activity": False,
                "max_workers": 16,
            },
            daemon=True,
        )
        ping_thread.start()
        ping_thread.join()
        stop_rain.set()
        rain_thread.join()
        clear_screen()
    else:
        update_pings(verify_api=False, update_activity=False, max_workers=16)

    if MODE_OVERRIDE_ERROR:
        print(f"{RED}{MODE_OVERRIDE_ERROR}{RESET}")
    elif MODE_OVERRIDE is not None and DEBUG_MODE:
        label = MODE_LABELS.get(MODE_OVERRIDE, MODE_OVERRIDE)
        print(f"{CYAN}Mode override: {label}{RESET}")

    forced_mode = MODE_OVERRIDE
    use_cli_mode = forced_mode is not None

    while True:
        if forced_mode is not None:
            mode = forced_mode
            forced_mode = None
        else:
            mode = choose_mode()
        if not mode or mode == "exit":
            break
        dispatch_mode(mode)
        if use_cli_mode:
            break
