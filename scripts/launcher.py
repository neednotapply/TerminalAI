#!/usr/bin/env python3
"""Simple launcher for TerminalAI utilities with cross-platform navigation."""
import os
import subprocess
import sys

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


def run_verbose() -> int | None:
    """Display options and read numeric choice."""
    for line in HEADER_LINES:
        print(line)
    for i, opt in enumerate(OPTIONS, 1):
        print(f"{i}) {opt}")
    try:
        choice = input("> ").strip()
    except EOFError:
        return None
    return int(choice) - 1 if choice in {"1", "2"} else None


def run_windows_menu() -> int | None:
    """Interactive menu using msvcrt for Windows."""
    idx = 0
    while True:
        os.system("cls")
        for line in HEADER_LINES:
            print(line)
        for i, opt in enumerate(OPTIONS):
            prefix = "> " if i == idx else "  "
            print(prefix + opt)
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
    return interactive_menu(header, OPTIONS)


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
