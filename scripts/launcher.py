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


def print_options(box_top: int, box_left: int) -> None:
    print(f"\033[{box_top + 2};{box_left + 3}H{GREEN}1) Start TerminalAI", end="")
    print(f"\033[{box_top + 3};{box_left + 3}H2) Scan Shodan{RESET}", end="")


def read_choice() -> str:
    if os.name == "nt":
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("1", "2"):
                    return ch
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
                        return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


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
        print_options(box_top, box_left)
        sys.stdout.flush()
        choice = read_choice()
    finally:
        stop_event.set()
        rain_thread.join()
        print("\033[0m\033[2J\033[H\033[?25h", end="")
        sys.stdout.flush()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if choice == "1":
        subprocess.call([sys.executable, os.path.join(script_dir, "TerminalAI.py")])
    else:
        subprocess.call([sys.executable, os.path.join(script_dir, "shodanscan.py")])


if __name__ == "__main__":
    main()
