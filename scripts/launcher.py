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
    from rain import rain
else:
    import curses
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
    try:
        while True:
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
    finally:
        stop_event.set()
        rain_thread.join()


def run_unix_menu() -> int | None:
    """Use curses-based menu on Unix-like systems with boxed logo and menu."""
    columns, rows = shutil.get_terminal_size(fallback=(80, 24))
    header_w = max(len(line) for line in HEADER_LINES) + 2
    header_h = len(HEADER_LINES) + 2
    header_x = max((columns - header_w) // 2, 0)

    menu_w = max(len("> " + o) for o in OPTIONS) + 2
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

    def _menu(stdscr):
        curses.curs_set(0)
        curses.start_color()
        try:
            if curses.can_change_color():
                curses.init_color(10, 20, 976, 0)
                curses.init_pair(1, 10, curses.COLOR_BLACK)
            else:
                curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        except curses.error:
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        green = curses.color_pair(1)
        idx = 0
        while True:
            stdscr.erase()
            _, cols = stdscr.getmaxyx()
            # draw header box at top
            header_win = curses.newwin(header_h, header_w, 0, max((cols - header_w) // 2, 0))
            header_win.attron(green)
            header_win.border()
            header_win.attroff(green)
            for i, line in enumerate(HEADER_LINES, 1):
                header_win.addstr(i, (header_w - len(line)) // 2, line, green)
            header_win.refresh()

            # draw menu box centered
            menu_win = curses.newwin(menu_h, menu_w, menu_y, menu_x)
            menu_win.attron(green)
            menu_win.border()
            menu_win.attroff(green)
            for i, opt in enumerate(OPTIONS):
                marker = "> " if i == idx else "  "
                text = marker + opt
                attr = curses.A_BOLD if i == idx else curses.A_NORMAL
                menu_win.addstr(1 + i, 1, text.ljust(menu_w - 2), green | attr)
            menu_win.refresh()

            key = stdscr.getch()
            if key == curses.KEY_UP:
                idx = (idx - 1) % len(OPTIONS)
            elif key == curses.KEY_DOWN:
                idx = (idx + 1) % len(OPTIONS)
            elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
                return idx
            elif key == 27:
                return None

    try:
        return curses.wrapper(_menu)
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
    if choice == 1:
        os.system("cls" if os.name == "nt" else "clear")
    subprocess.call([sys.executable, os.path.join(script_dir, target), *args])


if __name__ == "__main__":
    main()
