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

HEADER_LINES = [
    "████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗      █████╗ ██╗",
    "╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║     ██╔══██╗██║",
    "   ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║     ███████║██║",
    "   ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║     ██╔══██║██║",
    "   ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗██║  ██║██║",
    "   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝",
]

MENU_OPTIONS = [
    {
        "label": "Chat with Ollama",
        "script": "TerminalAI.py",
        "extra_args": ["--mode", "chat"],
    },
    {
        "label": "Generate Images with InvokeAI",
        "script": "TerminalAI.py",
        "extra_args": ["--mode", "image"],
    },
    {
        "label": "Scan for Servers using Shodan",
        "script": "shodanscan.py",
    },
]

OPTIONS = [opt["label"] for opt in MENU_OPTIONS]
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

        selected = MENU_OPTIONS[choice]
        if selected["script"] == "shodanscan.py" or choice != 0:
            os.system("cls" if os.name == "nt" else "clear")

        cmd = [sys.executable, os.path.join(script_dir, selected["script"])]
        cmd.extend(selected.get("extra_args", []))
        cmd.extend(args)

        try:
            subprocess.call(cmd)
        finally:
            if not DEBUG_MODE:
                os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    main()
