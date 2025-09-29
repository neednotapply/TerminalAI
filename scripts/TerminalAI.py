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
from typing import Any, Dict, List, Optional
from pathlib import Path
from rain import rain
from invoke_client import (
    DEFAULT_CFG_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_SCHEDULER,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    InvokeAIClient,
    InvokeAIClientError,
)

if os.name == "nt":
    import msvcrt
else:
    import curses
    import curses.textpad
    import termios
    import tty

# ANSI colors
GREEN = "\033[38;2;5;249;0m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
AI_COLOR = "\033[32m"

# Seconds to keep the "Hack The Planet" splash visible before continuing.
CONNECTING_SCREEN_DURATION = 0.5

VERBOSE = "--verbose" in sys.argv

MODE_ALIASES = {
    "chat": 0,
    "llm": 0,
    "image": 1,
    "img": 1,
    "invoke": 1,
    "invokeai": 1,
}
MODE_LABELS = {0: "Chat with LLM", 1: "Generate Image (InvokeAI)"}
MODE_OVERRIDE = None
MODE_OVERRIDE_ERROR = None


def _parse_mode_override():
    """Consume --mode arguments from sys.argv and configure overrides."""

    global MODE_OVERRIDE, MODE_OVERRIDE_ERROR
    args = sys.argv[1:]
    cleaned = [sys.argv[0]]
    i = 0
    while i < len(args):
        arg = args[i]
        value = None
        if arg == "--mode":
            if i + 1 < len(args):
                value = args[i + 1]
                i += 2
            else:
                MODE_OVERRIDE_ERROR = "--mode flag requires a value"
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
            MODE_OVERRIDE_ERROR = "--mode flag requires a value"
            MODE_OVERRIDE = None
        elif key in MODE_ALIASES:
            MODE_OVERRIDE = MODE_ALIASES[key]
            MODE_OVERRIDE_ERROR = None
        else:
            MODE_OVERRIDE_ERROR = f"Unknown mode '{value}'"
            MODE_OVERRIDE = None

    sys.argv[:] = cleaned


_parse_mode_override()

def clear_screen(force=False):
    if not VERBOSE or force:
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
CSV_PATHS = {
    "ollama": OLLAMA_CSV_PATH,
    "invokeai": INVOKE_CSV_PATH,
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
}
CONV_DIR = DATA_DIR / "conversations"
LOG_DIR = DATA_DIR / "logs"

MODEL_CAPABILITIES: Dict[str, Dict[str, Any]] = {}
EMBED_MODEL_KEYWORDS = ("embed", "embedding")


def _has_embedding_keyword(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(keyword in lowered for keyword in EMBED_MODEL_KEYWORDS)


def update_model_capabilities_from_metadata(name: str, metadata: Dict[str, Any]) -> None:
    info = MODEL_CAPABILITIES.setdefault(name, {})
    if info.get("is_embedding"):
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
        info["is_embedding"] = True
        info["confirmed"] = True
        info.pop("inferred", None)


def mark_model_as_embedding(name: str) -> None:
    info = MODEL_CAPABILITIES.setdefault(name, {})
    info["is_embedding"] = True
    info["confirmed"] = True
    info.pop("inferred", None)


def is_embedding_model(name: str) -> bool:
    info = MODEL_CAPABILITIES.get(name)
    if info and info.get("is_embedding"):
        return True
    if _has_embedding_keyword(name):
        info = MODEL_CAPABILITIES.setdefault(name, {})
        info["is_embedding"] = True
        info.setdefault("inferred", True)
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


def interactive_menu(header, options):
    idx = 0
    offset = 0
    while True:
        clear_screen()
        print(f"{GREEN}{header}{RESET}")
        rows = shutil.get_terminal_size(fallback=(80, 24)).lines
        view_height = max(1, rows - 2)
        end = offset + view_height
        visible = options[offset:end]
        for i, opt in enumerate(visible):
            actual = offset + i
            marker = f"{GREEN}> {RESET}" if actual == idx else "  "
            line = (
                f"{BOLD}{GREEN}{opt}{RESET}" if actual == idx else f"{GREEN}{opt}{RESET}"
            )
            print(f"{marker}{line}")
        key = get_key()
        if key == "UP":
            idx = (idx - 1) % len(options)
        elif key == "DOWN":
            idx = (idx + 1) % len(options)
        elif key in ("ENTER", "SPACE"):
            return idx
        elif key == "ESC":
            return None
        if idx < offset:
            offset = idx
        elif idx >= offset + view_height:
            offset = idx - view_height + 1


if os.name != "nt":
    from curses_nav import get_input as curses_get_input, interactive_menu as curses_menu
    get_input = curses_get_input
    interactive_menu = curses_menu


def display_with_chafa(image_path):
    if os.name == "nt":
        print(f"{YELLOW}Preview not supported on Windows. Image saved to {image_path}{RESET}")
        return
    try:
        result = subprocess.run(["chafa", str(image_path)], check=False)
        if result.returncode not in (0, 1):
            print(f"{YELLOW}chafa returned code {result.returncode}. Image saved to {image_path}{RESET}")
    except FileNotFoundError:
        print(
            f"{YELLOW}chafa not installed. Install from https://hpjansson.org/chafa/ to preview images.{RESET}"
        )


def confirm_exit():
    choice = interactive_menu("Exit?", ["No", "Yes"])
    return choice == 1


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


def choose_mode():
    options = ["Chat with LLM", "Generate Image (InvokeAI)", "Exit"]
    if VERBOSE:
        print(f"{CYAN}Mode Selection:{RESET}")
        for idx, opt in enumerate(options, 1):
            print(f"{GREEN}{idx}. {opt}{RESET}")
        while True:
            choice = get_input(f"{CYAN}Choose an option: {RESET}")
            if choice == "ESC":
                return 2
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return int(choice) - 1
            print(f"{RED}Invalid selection{RESET}")
    else:
        sel = interactive_menu("Select Mode:", options)
        return 2 if sel is None else sel


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


def update_pings(target_api=None):
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
        for idx, row in df.iterrows():
            if not normalise_bool(row.get("is_active", True)):
                continue
            latency = ping_time(row["ip"], row["port"])
            if latency is None:
                df.at[idx, "ping"] = pd.NA
                df.at[idx, "is_active"] = False
                if "inactive_reason" in df.columns:
                    df.at[idx, "inactive_reason"] = "ping timeout"
            else:
                if api == "invokeai":
                    api_ok = check_invoke_api(row["ip"], row["port"])
                else:
                    api_ok = check_ollama_api(row["ip"], row["port"])
                df.at[idx, "ping"] = latency
                df.at[idx, "is_active"] = api_ok
                if "inactive_reason" in df.columns:
                    df.at[idx, "inactive_reason"] = "" if api_ok else "api unreachable"
            if "last_check_date" in df.columns:
                df.at[idx, "last_check_date"] = now
        write_endpoints(api, df)

def select_server(servers, allow_back=False):
    if VERBOSE:
        print(f"{CYAN}Available Servers:{RESET}")
        for i, s in enumerate(servers, 1):
            ping_val = s.get("ping", float("inf"))
            ping_str = "?" if ping_val == float("inf") else f"{ping_val:.1f} ms"
            print(f"{GREEN}{i}. {s['nickname']} ({s['ip']}) - {ping_str}{RESET}")
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
            if c.isdigit() and 1 <= int(c) <= len(servers):
                return servers[int(c) - 1]
            print(f"{RED}Invalid selection{RESET}")
    else:
        options = []
        for s in servers:
            ping_val = s.get("ping", float("inf"))
            ping_str = "?" if ping_val == float("inf") else f"{ping_val:.1f} ms"
            options.append(f"{s['nickname']} ({s['ip']}) - {ping_str}")
        while True:
            choice = interactive_menu("Available Servers:", options)
            if choice is None:
                if allow_back:
                    return None
                if confirm_exit():
                    sys.exit(0)
                else:
                    continue
            return servers[choice]

def select_model(models):
    if VERBOSE:
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
    if VERBOSE:
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

    if VERBOSE:
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
                        if not VERBOSE:
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
                            if not VERBOSE:
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
            used_embeddings = False
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
                    embed_resp, embed_failed, embed_timeout = try_embeddings_request(
                        model, user_input, headers, timeout_val
                    )
                    if embed_resp:
                        resp = embed_resp
                        used_embeddings = True
                        server_failed = False
                    server_failed = server_failed or embed_failed
                    timed_out = timed_out or embed_timeout

                if not resp:
                    print(f"{RED}[Error] No response{RESET}")
                    if not server_failed:
                        server_failed = True

            elapsed = stop_thinking_timer(start, stop_event, timed_out)

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
                if used_embeddings:
                    history.append({"user": user_input, "ai": resp, "elapsed": elapsed})
                else:
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
        servers = load_servers("ollama")
        srv_list = [s for s in servers if selected_api in s["apis"]]
        if not srv_list:
            print(f"{RED}No active Ollama servers available{RESET}")
            get_input(f"{CYAN}Press Enter to return to the main menu{RESET}")
            return

        choice = select_server(srv_list, allow_back=True)
        if choice is None:
            return
        selected_server = choice
        clear_screen()
        SERVER_URL = build_url(selected_server, selected_api)

        models = fetch_models()
        if not models:
            print(f"{RED}No models found on selected server{RESET}")
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
            chosen = select_model(models)
            if chosen is None:
                clear_screen()
                break
            clear_screen()
            while True:
                if has_conversations(chosen):
                    conv_file, messages, history, context = select_conversation(chosen)
                    if conv_file == "back":
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
                    break
                if result == "server_inactive":
                    conv_file = "server_inactive"
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
            break


def run_image_mode():
    global SERVER_URL, selected_server, selected_api
    selected_api = "invokeai"
    while True:
        clear_screen()
        servers = load_servers("invokeai")
        if not servers:
            print(f"{RED}No InvokeAI servers available{RESET}")
            get_input(f"{CYAN}Press Enter to return to the main menu{RESET}")
            return

        server_choice = select_server(servers, allow_back=True)
        if server_choice is None:
            return
        selected_server = server_choice
        port = selected_server["apis"].get("invokeai")
        if port is None:
            print(f"{RED}Selected server does not expose InvokeAI API{RESET}")
            get_input(f"{CYAN}Press Enter to pick another server{RESET}")
            continue

        client = InvokeAIClient(selected_server["ip"], port, selected_server["nickname"], DATA_DIR)
        try:
            models = client.list_models()
        except InvokeAIClientError as exc:
            print(f"{RED}{exc}{RESET}")
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
            print(f"{RED}No InvokeAI models reported by this server{RESET}")
            get_input(f"{CYAN}Press Enter to pick another server{RESET}")
            continue

        while True:
            model = select_invoke_model(models)
            if model is None:
                break
            while True:
                prompt = get_input(f"{CYAN}Prompt: {RESET}")
                if prompt == "ESC":
                    break
                if not prompt.strip():
                    print(f"{RED}Prompt cannot be empty{RESET}")
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
                scheduler = get_input(f"{CYAN}Scheduler [{DEFAULT_SCHEDULER}]: {RESET}")
                if scheduler == "ESC":
                    break
                scheduler = scheduler.strip() or DEFAULT_SCHEDULER
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
                        continue

                width = max(64, (width // 8) * 8)
                height = max(64, (height // 8) * 8)

                try:
                    result = client.generate_image(
                        model=model,
                        prompt=prompt.strip(),
                        negative_prompt=(negative or "").strip(),
                        width=width,
                        height=height,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        scheduler=scheduler,
                        seed=seed_val,
                    )
                except InvokeAIClientError as exc:
                    print(f"{RED}{exc}{RESET}")
                    get_input(f"{CYAN}Press Enter to try again{RESET}")
                    continue
                except requests.RequestException as exc:
                    print(f"{RED}Network error: {exc}{RESET}")
                    get_input(f"{CYAN}Press Enter to try again{RESET}")
                    continue

                image_path = result["path"]
                print(f"{GREEN}Image saved to {image_path}{RESET}")
                metadata_path = result.get("metadata_path")
                if metadata_path:
                    print(f"{CYAN}Metadata saved to {metadata_path}{RESET}")
                display_with_chafa(image_path)

                keep = get_input(f"{CYAN}Keep image? (y/n) [y]: {RESET}")
                if keep == "ESC":
                    keep = "y"
                if keep and keep.lower().startswith("n"):
                    try:
                        image_path.unlink(missing_ok=True)
                        if metadata_path:
                            Path(metadata_path).unlink(missing_ok=True)
                        print(f"{YELLOW}Image discarded.{RESET}")
                    except Exception as exc:
                        print(f"{RED}Failed to delete image: {exc}{RESET}")
                else:
                    print(f"{GREEN}Image retained.{RESET}")

                cont = get_input(
                    f"{CYAN}Press Enter to generate again, type 'back' to change model, or ESC for server menu: {RESET}"
                )
                if cont == "ESC":
                    return
                if cont.strip().lower() == "back":
                    break
            break


# Main loop
if __name__ == "__main__":
    selected_api = "ollama"

    if not VERBOSE:
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
        ping_thread = threading.Thread(target=update_pings, daemon=True)
        ping_thread.start()
        ping_thread.join()
        stop_rain.set()
        rain_thread.join()
        clear_screen()
    else:
        update_pings()

    if MODE_OVERRIDE_ERROR:
        print(f"{RED}{MODE_OVERRIDE_ERROR}{RESET}")
    elif MODE_OVERRIDE is not None and VERBOSE:
        label = MODE_LABELS.get(MODE_OVERRIDE)
        if label:
            print(f"{CYAN}Mode override: {label}{RESET}")

    forced_mode = MODE_OVERRIDE
    use_cli_mode = forced_mode is not None

    while True:
        if forced_mode is not None:
            mode = forced_mode
            forced_mode = None
        else:
            mode = choose_mode()
        if mode == 0:
            run_chat_mode()
        elif mode == 1:
            run_image_mode()
        else:
            break
        if use_cli_mode:
            break
