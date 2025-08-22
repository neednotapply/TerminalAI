#!/usr/bin/env python3
"""Simple launcher for TerminalAI utilities with cross-platform navigation."""
import os
import subprocess
import sys
import shutil
import threading

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

OPTIONS = ["Start TerminalAI", "Scan Shodan"]
VERBOSE = "--verbose" in sys.argv

if os.name == "nt":
    import msvcrt
else:
    from curses_nav import interactive_menu
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
    return int(choice) - 1 if choice in {"1", "2"} else None


def run_windows_menu() -> int | None:
    """Interactive menu using msvcrt for Windows."""
    idx = 0
    columns, _ = shutil.get_terminal_size(fallback=(80, 24))
    while True:
        os.system("cls")
        for line in HEADER_LINES:
            print(f"{GREEN}{line.center(columns)}{RESET}")
        for i, opt in enumerate(OPTIONS):
            prefix = "> " if i == idx else "  "
            text = prefix + opt
            print(f"{GREEN}{text.center(columns)}{RESET}")
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            return idx
        if ch == "\x1b":
            return None
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            if ch2 == "H":
                idx = (idx - 1) % len(OPTIONS)
            elif ch2 == "P":
                idx = (idx + 1) % len(OPTIONS)


def run_unix_menu() -> int | None:
    """Use curses-based menu on Unix-like systems."""
    header = "\n".join(HEADER_LINES)
    columns, _ = shutil.get_terminal_size(fallback=(80, 24))
    max_width = max(len(line) for line in HEADER_LINES + [f"> {o}" for o in OPTIONS])
    start_col = max((columns - max_width) // 2, 0) + 1
    stop_event = threading.Event()
    rain_thread = threading.Thread(
        target=rain,
        kwargs={
            "persistent": True,
            "stop_event": stop_event,
            "box_top": 1,
            "box_bottom": len(HEADER_LINES) + len(OPTIONS),
            "box_left": start_col,
            "box_right": start_col + max_width - 1,
            "clear_screen": False,
        },
        daemon=True,
    )
    rain_thread.start()
    try:
        return interactive_menu(header, OPTIONS, center=True)
    finally:
        stop_event.set()
        rain_thread.join()


def main() -> None:
    args = sys.argv[1:]
    if VERBOSE:
        choice = run_verbose()
    else:
        choice = run_windows_menu() if os.name == "nt" else run_unix_menu()
    if choice is None:
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target = "TerminalAI.py" if choice == 0 else "shodanscan.py"
    subprocess.call([sys.executable, os.path.join(script_dir, target), *args])


if __name__ == "__main__":
    main()
