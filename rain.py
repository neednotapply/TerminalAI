#!/usr/bin/env python3
import argparse
import random
import shutil
import sys
import time

GREEN = "\033[32m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header-top", type=int, default=0)
    parser.add_argument("--header-bottom", type=int, default=-1)
    parser.add_argument("--box-top", type=int, default=0)
    parser.add_argument("--box-bottom", type=int, default=-1)
    parser.add_argument("--box-left", type=int, default=0)
    parser.add_argument("--box-right", type=int, default=-1)
    parser.add_argument("--prompt-row", type=int, default=0)
    args = parser.parse_args()

    cols, rows = shutil.get_terminal_size()
    chars = "01"
    try:
        while True:
            r = random.randint(0, max(args.prompt_row - 1, 0))
            c = random.randint(0, cols - 1)
            if args.header_top <= r <= args.header_bottom:
                continue
            if (
                args.box_top <= r <= args.box_bottom
                and args.box_left <= c <= args.box_right
            ):
                continue
            sys.stdout.write(f"\033[{r};{c}H{GREEN}{random.choice(chars)}{RESET}")
            sys.stdout.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
