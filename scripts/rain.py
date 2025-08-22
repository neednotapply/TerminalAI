#!/usr/bin/env python3
import argparse
import os
import random
import select
import shutil
import sys
import time

RESET = "\033[0m"
GREEN = "\033[38;2;5;249;0m"

if os.name == "nt":
    import msvcrt
else:
    import termios
    import tty


def rain(
    persistent=False,
    duration=3,
    stop_event=None,
    box_top=None,
    box_bottom=None,
    box_left=None,
    box_right=None,
    boxes=None,
    clear_screen=True,
):
    if boxes is None:
        boxes = []
    if all(v is not None for v in (box_top, box_bottom, box_left, box_right)):
        boxes.append(
            {
                "top": box_top,
                "bottom": box_bottom,
                "left": box_left,
                "right": box_right,
            }
        )
    charset = (
        "01$#*+=-░▒▓▌▐▄▀▁▂▃▅▆▇█ﾊﾐﾋｰｱｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ" "01$#*+=-"
    )
    fd = None
    old_settings = None
    stdin_is_tty = sys.stdin.isatty()
    try:
        print("\033[?25l", end="")
        if clear_screen:
            os.system("cls" if os.name == "nt" else "clear")
        end_time = time.time() + duration if not persistent else None
        columns, rows = shutil.get_terminal_size(fallback=(80, 24))
        trail_length = 6
        drops = [random.randint(0, rows) for _ in range(columns)]
        if os.name != "nt" and stdin_is_tty:
            fd = sys.stdin.fileno()
            try:
                old_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)
            except termios.error:
                stdin_is_tty = False
        while (persistent or time.time() < end_time) and (
            stop_event is None or not stop_event.is_set()
        ):
            for col in range(columns):
                drop_pos = drops[col]
                for t in range(trail_length):
                    y = drop_pos - t
                    if 0 < y < rows:
                        if any(
                            box["top"] <= y <= box["bottom"]
                            and box["left"] <= col + 1 <= box["right"]
                            for box in boxes
                        ):
                            continue
                        char = random.choice(charset) if t < 4 else " "
                        shade = (
                            GREEN
                            if t == 0
                            else "\033[32m"
                            if t == 1
                            else "\033[32;2m"
                            if t == 2
                            else "\033[2;32m"
                            if t == 3
                            else ""
                        )
                        print(
                            f"\033[{y};{col+1}H{shade}{char}{RESET}",
                            end="",
                        )
                drops[col] = (drops[col] + 1) % (rows + trail_length)
            sys.stdout.flush()
            time.sleep(0.08)
            if stop_event is not None and stop_event.is_set():
                break
            if persistent and (stop_event is None or not stop_event.is_set()):
                if os.name == "nt":
                    if msvcrt.kbhit():
                        msvcrt.getch()
                        break
                elif stdin_is_tty:
                    dr, _, _ = select.select([sys.stdin], [], [], 0)
                    if dr:
                        os.read(sys.stdin.fileno(), 1)
                        break
    finally:
        if os.name != "nt" and old_settings is not None and fd is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        time.sleep(0.5)
        if clear_screen:
            print("\033[0m\033[2J\033[H\033[?25h", end="")
        else:
            print("\033[0m\033[?25h", end="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument(
        "--exclude",
        action="append",
        metavar="TOP,BOTTOM,LEFT,RIGHT",
        help="Exclusion box coordinates",
    )
    parser.add_argument("--no-clear", action="store_true", help="Don't clear screen")
    args = parser.parse_args()
    boxes = []
    if args.exclude:
        for ex in args.exclude:
            t, b, l, r = map(int, ex.split(","))
            boxes.append({"top": t, "bottom": b, "left": l, "right": r})
    rain(
        persistent=args.persistent,
        duration=args.duration,
        boxes=boxes,
        clear_screen=not args.no_clear,
    )


if __name__ == "__main__":
    main()
