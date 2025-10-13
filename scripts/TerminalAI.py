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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from pathlib import Path
from rain import rain
from invoke_client import (
    DEFAULT_CFG_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_SCHEDULER,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    UNCATEGORIZED_BOARD_ID,
    InvokeAIClient,
    InvokeAIClientError,
    InvokeAIModel,
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
    "chat": 0,
    "llm": 0,
    "image": 1,
    "img": 1,
    "invoke": 1,
    "invokeai": 1,
}
MODE_LABELS = {0: "Chat with Ollama", 1: "Generate Images with InvokeAI"}
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


def _cleanup_image_result(result):
    path = result.get("path") if isinstance(result, dict) else None
    metadata_path = result.get("metadata_path") if isinstance(result, dict) else None

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
        if path_value:
            print(f"{CYAN}Preview saved to {path_value}{RESET}")
        if isinstance(metadata, dict):
            _print_board_image_summary(metadata)
        if path_value:
            display_with_chafa(path_value)

        saved = False
        current_path = Path(path_value) if path_value else None
        metadata_path_value = result.get("metadata_path") if isinstance(result, dict) else None

        while True:
            print(
                f"{CYAN}\u2190/A for previous, \u2192/D for next, Enter to save, Esc to return.{RESET}"
            )
            action = get_key()
            if action in ("a", "A"):
                action = "LEFT"
            elif action in ("d", "D"):
                action = "RIGHT"

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
                                result["metadata_path"] = str(target_meta)
                        except OSError as exc:
                            print(f"{YELLOW}Failed to rename metadata file: {exc}{RESET}")
                saved = True
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
                if not saved:
                    _cleanup_image_result(result)
                print(f"{CYAN}Loading next image...{RESET}")
                index += 1
                break

            if action == "LEFT":
                if index == 0:
                    print(f"{YELLOW}Already viewing the oldest image loaded for this board.{RESET}")
                    continue
                if not saved:
                    _cleanup_image_result(result)
                print(f"{CYAN}Loading previous image...{RESET}")
                index -= 1
                break

            if action == "ESC":
                if not saved:
                    _cleanup_image_result(result)
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
            _cleanup_image_result(preview)
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
        model = select_invoke_model(models)
        if model is None:
            return

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
                    f"{YELLOW}Server did not return a queue item id. Check the Auto board for the finished image.{RESET}"
                )
            if batch_id:
                print(f"{CYAN}Batch id: {batch_id}{RESET}")
            if seed_used is not None:
                print(f"{CYAN}Seed used: {seed_used}{RESET}")

            acknowledgement = get_input(
                f"{CYAN}Press Enter to prompt again (ESC=change model): {RESET}"
            )
            if isinstance(acknowledgement, str) and acknowledgement.strip().lower() == "esc":
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
    scheduler_name = (scheduler or "").strip() or DEFAULT_SCHEDULER

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
    options = ["Chat with Ollama", "Generate Images with InvokeAI", "Exit"]
    if DEBUG_MODE:
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


def _probe_endpoint(api_type, ip, port, perform_api_check):
    latency = ping_time(ip, port)
    api_ok = None
    if latency is not None and perform_api_check:
        if api_type == "invokeai":
            api_ok = check_invoke_api(ip, port)
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


def select_scheduler_option(options: List[str], default_value: str) -> Optional[str]:
    """Prompt for a scheduler using free text input."""

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

    while True:
        entry = get_input(f"{CYAN}Scheduler [{default_value}]: {RESET}")
        if entry == "ESC":
            return None
        entry = entry.strip()
        if not entry:
            return default_value
        return entry


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
            board_id = client.ensure_board(TERMINALAI_BOARD_NAME)
        except InvokeAIClientError as exc:
            print(f"{RED}{exc}{RESET}")
            get_input(f"{CYAN}Press Enter to pick another server{RESET}")
            continue
        if not _is_valid_terminalai_board_id(board_id):
            print(f"{RED}{TERMINALAI_BOARD_ID_ERROR_MESSAGE}{RESET}")
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
            print(f"{YELLOW}This server did not return any models. Image generation may be unavailable.{RESET}")
            get_input(f"{CYAN}Press Enter to continue{RESET}")
            clear_screen()

        while True:
            options = ["View Boards", "Generate Images", "Rename Server", "[Back]"]
            header = f"InvokeAI on {client.nickname}:"
            choice = interactive_menu(header, options)
            if choice is None or choice == len(options) - 1:
                break
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
