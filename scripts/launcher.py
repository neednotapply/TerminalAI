#!/usr/bin/env python3
"""Simple launcher for TerminalAI utilities with cross-platform navigation."""
import os
import subprocess
import sys
import shutil
import threading
import time

GREEN = "\033[38;2;5;249;0m"
RESET = "\033[0m"
BOLD = "\033[1m"

HEADER_LINES = [
    "████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗      █████╗ ██╗",
    "╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║     ██╔══██╗██║",
    "   ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║     ███████║██║",
    "   ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║     ██╔══██║██║",
    "   ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗██║  ██║██║",
    "   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝",
]

TOP_LEVEL_OPTIONS = [
    {"label": "LLM Chat", "key": "llm"},
    {"label": "Image Generation", "key": "image"},
    {"label": "Shodan Scan", "key": "shodan"},
    {"label": "[Exit]", "key": "exit"},
]

PROVIDER_OPTIONS = {
    "llm": [
        {
            "label": "Ollama",
            "script": "TerminalAI.py",
            "extra_args": ["--mode", "llm-ollama"],
            "clear_before": False,
        },
        {"label": "[Back]", "key": "back"},
    ],
    "image": [
        {
            "label": "InvokeAI",
            "script": "TerminalAI.py",
            "extra_args": ["--mode", "image-invokeai"],
        },
        {
            "label": "Automatic1111",
            "script": "TerminalAI.py",
            "extra_args": ["--mode", "image-automatic1111"],
        },
        {"label": "[Back]", "key": "back"},
    ],
}

PROVIDER_HEADERS = {
    "llm": "Select LLM Provider:",
    "image": "Select Image Generation Provider:",
}

DIRECT_ACTIONS = {
    "shodan": {"label": "Shodan Scan", "script": "shodanscan.py"},
}

OPTIONS = [opt["label"] for opt in TOP_LEVEL_OPTIONS]
DEBUG_FLAGS = {"--debug", "--verbose"}
DEBUG_MODE = any(flag in sys.argv for flag in DEBUG_FLAGS)

if os.name == "nt":
    import msvcrt
    from rain import rain
else:
    import termios
    import tty
    import select
    from rain import rain


def run_verbose() -> int | None:
    """Display options and read numeric choice."""
    for line in HEADER_LINES:
        print(f"{GREEN}{line}{RESET}")
    for i, opt in enumerate(OPTIONS, 1):
        print(f"{GREEN}{i}) {opt}{RESET}")
    try:
        choice = input(f"{GREEN}> {RESET}").strip()
    except EOFError:
        return None
    if not choice.isdigit():
        return None
    idx = int(choice) - 1
    return idx if 0 <= idx < len(OPTIONS) else None


def _arrow_menu(header: str, labels: list[str]) -> int | None:
    """Simple arrow-key menu without matrix rain effects."""

    if not labels:
        return None

    if os.name == "nt":

        def read_key() -> str | None:
            while True:
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    return "ENTER"
                if ch == "\x1b":
                    return "ESC"
                if ch in ("\x00", "\xe0"):
                    ch2 = msvcrt.getwch()
                    if ch2 == "H":
                        return "UP"
                    if ch2 == "P":
                        return "DOWN"
                elif ch == " ":
                    return "ENTER"

        idx = 0
        while True:
            os.system("cls")
            print(f"{GREEN}{header}{RESET}")
            for i, label in enumerate(labels):
                prefix = "> " if i == idx else "  "
                style = BOLD if i == idx else ""
                print(f"{style}{GREEN}{prefix}{label}{RESET}")
            key = read_key()
            if key == "UP":
                idx = (idx - 1) % len(labels)
            elif key == "DOWN":
                idx = (idx + 1) % len(labels)
            elif key == "ENTER":
                return idx
            elif key == "ESC":
                return None

    else:

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        def read_key() -> str | None:
            while True:
                r, _, _ = select.select([sys.stdin], [], [], None)
                if not r:
                    continue
                ch = os.read(fd, 1).decode()
                if not ch:
                    return None
                if ch in ("\n", "\r"):
                    return "ENTER"
                if ch == " ":
                    return "ENTER"
                if ch == "\x1b":
                    r2, _, _ = select.select([sys.stdin], [], [], 0.01)
                    if not r2:
                        return "ESC"
                    ch2 = os.read(fd, 1).decode()
                    if ch2 == "[":
                        ch3 = os.read(fd, 1).decode()
                        if ch3 == "A":
                            return "UP"
                        if ch3 == "B":
                            return "DOWN"
                    return "ESC"

        idx = 0
        try:
            while True:
                os.system("clear")
                print(f"{GREEN}{header}{RESET}")
                for i, label in enumerate(labels):
                    prefix = "> " if i == idx else "  "
                    style = BOLD if i == idx else ""
                    print(f"{style}{GREEN}{prefix}{label}{RESET}")
                key = read_key()
                if key == "UP":
                    idx = (idx - 1) % len(labels)
                elif key == "DOWN":
                    idx = (idx + 1) % len(labels)
                elif key == "ENTER":
                    return idx
                elif key == "ESC":
                    return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return None


def run_windows_menu() -> int | None:
    """Interactive menu using msvcrt for Windows with boxed layout."""
    columns, rows = shutil.get_terminal_size(fallback=(80, 24))
    header_w = max(len(line) for line in HEADER_LINES) + 2
    header_x = max((columns - header_w) // 2, 0)
    header_h = len(HEADER_LINES) + 2

    menu_w = max(len(opt) + 2 for opt in OPTIONS) + 2
    menu_h = len(OPTIONS) + 2
    menu_x = max((columns - menu_w) // 2, 0)
    menu_y = max((rows - menu_h) // 2, 0)

    boxes = [
        {"top": 1, "bottom": header_h, "left": header_x + 1, "right": header_x + header_w},
        {
            "top": menu_y + 1,
            "bottom": menu_y + menu_h,
            "left": menu_x + 1,
            "right": menu_x + menu_w,
        },
    ]
    stop_event = threading.Event()
    rain_thread = threading.Thread(
        target=rain,
        kwargs={"persistent": True, "stop_event": stop_event, "boxes": boxes, "clear_screen": False},
        daemon=True,
    )
    rain_thread.start()

    idx = 0

    def draw_menu() -> None:
        os.system("cls")
        print(f"{' ' * header_x}{GREEN}┌{'─' * (header_w - 2)}┐{RESET}")
        for line in HEADER_LINES:
            print(f"{' ' * header_x}{GREEN}│{line.center(header_w - 2)}│{RESET}")
        print(f"{' ' * header_x}{GREEN}└{'─' * (header_w - 2)}┘{RESET}")

        for _ in range(max(0, menu_y - header_h)):
            print()
        print(f"{' ' * menu_x}{GREEN}┌{'─' * (menu_w - 2)}┐{RESET}")
        for i, opt in enumerate(OPTIONS):
            prefix = "> " if i == idx else "  "
            line = prefix + opt
            print(f"{' ' * menu_x}{GREEN}│{line.ljust(menu_w - 2)}│{RESET}")
        print(f"{' ' * menu_x}{GREEN}└{'─' * (menu_w - 2)}┘{RESET}")

    try:
        draw_menu()
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    return idx
                if ch == "\x1b":
                    return None
                if ch in ("\x00", "\xe0"):
                    ch2 = msvcrt.getwch()
                    if ch2 == "H":
                        idx = (idx - 1) % len(OPTIONS)
                        draw_menu()
                    elif ch2 == "P":
                        idx = (idx + 1) % len(OPTIONS)
                        draw_menu()
            else:
                time.sleep(0.05)
    finally:
        stop_event.set()
        rain_thread.join()


def run_unix_menu() -> int | None:
    """Interactive menu for Unix-like systems with matrix rain."""

    columns, rows = shutil.get_terminal_size(fallback=(80, 24))
    header_w = max(len(line) for line in HEADER_LINES) + 2
    header_x = max((columns - header_w) // 2, 0)
    header_h = len(HEADER_LINES) + 2

    menu_w = max(len(opt) + 2 for opt in OPTIONS) + 2
    menu_h = len(OPTIONS) + 2
    menu_x = max((columns - menu_w) // 2, 0)
    menu_y = max((rows - menu_h) // 2, 0)

    boxes = [
        {"top": 1, "bottom": header_h, "left": header_x + 1, "right": header_x + header_w},
        {
            "top": menu_y + 1,
            "bottom": menu_y + menu_h,
            "left": menu_x + 1,
            "right": menu_x + menu_w,
        },
    ]

    stop_event = threading.Event()
    rain_thread = threading.Thread(
        target=rain,
        kwargs={"persistent": True, "stop_event": stop_event, "boxes": boxes, "clear_screen": False},
        daemon=True,
    )
    rain_thread.start()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    idx = 0

    def draw_menu() -> None:
        os.system("clear")
        print(f"{' ' * header_x}{GREEN}┌{'─' * (header_w - 2)}┐{RESET}")
        for line in HEADER_LINES:
            print(f"{' ' * header_x}{GREEN}│{line.center(header_w - 2)}│{RESET}")
        print(f"{' ' * header_x}{GREEN}└{'─' * (header_w - 2)}┘{RESET}")

        for _ in range(max(0, menu_y - header_h)):
            print()
        print(f"{' ' * menu_x}{GREEN}┌{'─' * (menu_w - 2)}┐{RESET}")
        for i, opt in enumerate(OPTIONS):
            prefix = "> " if i == idx else "  "
            line = prefix + opt
            print(f"{' ' * menu_x}{GREEN}│{line.ljust(menu_w - 2)}│{RESET}")
        print(f"{' ' * menu_x}{GREEN}└{'─' * (menu_w - 2)}┘{RESET}")

    try:
        draw_menu()
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = os.read(fd, 1).decode()
                if ch in ("\n", "\r"):
                    return idx
                if ch == "\x1b":
                    r2, _, _ = select.select([sys.stdin], [], [], 0.0001)
                    if r2:
                        ch2 = os.read(fd, 1).decode()
                        if ch2 == "[":
                            ch3 = os.read(fd, 1).decode()
                            if ch3 == "A":
                                idx = (idx - 1) % len(OPTIONS)
                                draw_menu()
                            elif ch3 == "B":
                                idx = (idx + 1) % len(OPTIONS)
                                draw_menu()
                        else:
                            return None
                    else:
                        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        stop_event.set()
        rain_thread.join()

    return None


def prompt_provider_menu(category_key: str) -> dict | None:
    options = PROVIDER_OPTIONS.get(category_key, [])
    if not options:
        return None

    header = PROVIDER_HEADERS.get(category_key, "Select an option:")
    labels = [opt["label"] for opt in options]

    while True:
        if DEBUG_MODE:
            if not labels:
                return None
            print(f"{GREEN}{header}{RESET}")
            for idx, label in enumerate(labels, 1):
                print(f"{GREEN}{idx}) {label}{RESET}")
            try:
                choice = input(f"{GREEN}> {RESET}").strip()
            except EOFError:
                return None
            if not choice:
                return None
            if not choice.isdigit():
                print("Invalid selection. Please choose a valid option.")
                time.sleep(0.8)
                continue
            index = int(choice) - 1
        else:
            index = _arrow_menu(header, labels)
            if index is None:
                return None

        if 0 <= index < len(options):
            selected = options[index]
            if selected.get("key") == "back":
                return None
            return selected
        print("Invalid selection. Please choose a valid option.")
        time.sleep(0.8)


def run_selected_option(option: dict, args: list[str], script_dir: str) -> None:
    if not option:
        return

    cmd = [sys.executable, os.path.join(script_dir, option["script"])]
    cmd.extend(option.get("extra_args", []))
    cmd.extend(args)

    should_clear = option.get("clear_before", True)
    if should_clear and not DEBUG_MODE:
        os.system("cls" if os.name == "nt" else "clear")

    try:
        subprocess.call(cmd)
    finally:
        if not DEBUG_MODE:
            os.system("cls" if os.name == "nt" else "clear")


def main() -> None:
    args = sys.argv[1:]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    while True:
        if DEBUG_MODE:
            choice = run_verbose()
        else:
            choice = run_windows_menu() if os.name == "nt" else run_unix_menu()

        if choice is None:
            return

        top_option = TOP_LEVEL_OPTIONS[choice]
        key = top_option.get("key")

        if key == "exit":
            return

        if key in PROVIDER_OPTIONS:
            option = prompt_provider_menu(key)
            if option is None:
                continue
        else:
            option = DIRECT_ACTIONS.get(key)
            if option is None:
                continue

        run_selected_option(option, args, script_dir)


if __name__ == "__main__":
    main()
