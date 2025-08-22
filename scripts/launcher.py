#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import threading
import time
import select

import rain

GREEN = "\033[32m"
RESET = "\033[0m"
BOLD = "\033[1m"

VERBOSE = "--verbose" in sys.argv

if os.name == "nt":
    import colorama
    colorama.init()
    import msvcrt
else:
    import termios
    import tty


HEADER_LINES = [
    "████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗      █████╗ ██╗",
    "╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║     ██╔══██╗██║",
    "   ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║     ███████║██║",
    "   ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║     ██╔══██║██║",
    "   ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗██║  ██║██║",
    "   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝",
]


def draw_header(top: int, left: int) -> None:
    for idx, line in enumerate(HEADER_LINES):
        print(f"\033[{top + idx + 1};{left + 1}H{BOLD}{GREEN}{line}{RESET}", end="")


def draw_box(top: int, left: int, width: int, height: int) -> None:
    horiz = "─" * (width - 2)
    print(f"\033[{top + 1};{left + 1}H{GREEN}┌{horiz}┐", end="")
    for i in range(1, height - 1):
        print(f"\033[{top + 1 + i};{left + 1}H│\033[{top + 1 + i};{left + width}H│", end="")
    print(f"\033[{top + height};{left + 1}H└{horiz}┘{RESET}", end="")


def print_options(
    box_top: int,
    box_left: int,
    box_width: int,
    options,
    idx: int,
    offset: int,
    view_height: int,
) -> None:
    inner_width = box_width - 4
    for i in range(view_height):
        y = box_top + 2 + i
        pos = offset + i
        if pos < len(options):
            opt = options[pos]
            marker = "> " if pos == idx else "  "
            line = f"{BOLD}{opt}{RESET}" if pos == idx else opt
            content = f"{marker}{line}".ljust(inner_width)
            print(
                f"\033[{y};{box_left + 3}H{GREEN}{content}{RESET}",
                end="",
            )
        else:
            print(
                f"\033[{y};{box_left + 3}H{' ' * inner_width}",
                end="",
            )


def print_options_verbose(box_top: int, box_left: int, options) -> None:
    for i, opt in enumerate(options, 1):
        print(
            f"\033[{box_top + 1 + i};{box_left + 3}H{GREEN}{i}) {opt}{RESET}",
            end="",
        )


def _read_escape_sequence(initial_timeout: float = 0.25) -> str:
    """Read characters following an ESC to capture full arrow sequences."""
    seq = ""
    dr, _, _ = select.select([sys.stdin], [], [], initial_timeout)
    if dr:
        seq += sys.stdin.read(1)
        while True:
            dr, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not dr:
                break
            seq += sys.stdin.read(1)
    return seq

def read_choice() -> int | None:
    if os.name == "nt":
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("1", "2"):
                    return int(ch) - 1
                if ch == "\x1b":
                    return None
                if ch in ("\x00", "\xe0"):
                    msvcrt.getwch()
                    continue
            time.sleep(0.05)
    else:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                dr, _, _ = select.select([sys.stdin], [], [], 0.05)
                if dr:
                    ch = sys.stdin.read(1)
                    if ch in ("1", "2"):
                        return int(ch) - 1
                    if ch == "\x1b":
                        seq = _read_escape_sequence()
                        if seq.endswith(("A", "B", "C", "D")):
                            continue
                        return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


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


def interactive_choice(
    box_top: int, box_left: int, box_width: int, box_height: int, options
) -> int | None:
    idx = 0
    offset = 0
    view_height = box_height - 2
    while True:
        print_options(box_top, box_left, box_width, options, idx, offset, view_height)
        sys.stdout.flush()
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


def main() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    print("\033[?25l", end="")
    cols, rows = shutil.get_terminal_size(fallback=(80, 24))
    header_height = 6
    header_width = 76
    header_top = 0
    header_left = (cols - header_width) // 2
    header_bottom = header_top + header_height - 1
    box_width = 34
    box_height = 4
    box_top = header_bottom + 2
    box_left = (cols - box_width) // 2
    box_bottom = box_top + box_height - 1
    boxes = [
        {
            "top": header_top + 1,
            "bottom": header_bottom + 1,
            "left": header_left + 1,
            "right": header_left + header_width,
        },
        {
            "top": box_top + 1,
            "bottom": box_bottom + 1,
            "left": box_left + 1,
            "right": box_left + box_width,
        },
    ]
    stop_event = threading.Event()
    rain_thread = threading.Thread(
        target=rain.rain,
        kwargs={
            "persistent": True,
            "boxes": boxes,
            "clear_screen": False,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    rain_thread.start()
    try:
        draw_header(header_top, header_left)
        draw_box(box_top, box_left, box_width, box_height)
        options = ["Start TerminalAI", "Scan Shodan"]
        if VERBOSE:
            print_options_verbose(box_top, box_left, options)
            sys.stdout.flush()
            choice = read_choice()
        else:
            choice = interactive_choice(box_top, box_left, box_width, box_height, options)
    finally:
        stop_event.set()
        rain_thread.join()
        print("\033[0m\033[2J\033[H\033[?25h", end="")
        sys.stdout.flush()
    if choice is None:
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args = sys.argv[1:]
    if choice == 0:
        subprocess.call([sys.executable, os.path.join(script_dir, "TerminalAI.py"), *args])
    else:
        subprocess.call([sys.executable, os.path.join(script_dir, "shodanscan.py"), *args])


if __name__ == "__main__":
    main()
