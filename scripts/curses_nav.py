import curses
import curses.textpad


def get_input(prompt: str) -> str:
    """Prompt for input with basic line editing using curses."""

    def _inner(stdscr):
        curses.curs_set(1)
        curses.echo()
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
        stdscr.addstr(prompt, green)
        stdscr.refresh()
        win = curses.newwin(1, curses.COLS - len(prompt) - 1, 0, len(prompt))
        box = curses.textpad.Textbox(win)
        text = box.edit().strip()
        return text

    return curses.wrapper(_inner)


def interactive_menu(header: str, options: list, *, center: bool = False) -> int | None:
    """Simple vertical menu navigated with arrow keys."""

    normalized = []
    for opt in options:
        if isinstance(opt, dict):
            label = str(opt.get("label", ""))
            style = opt.get("style") or opt.get("color")
        elif isinstance(opt, (list, tuple)):
            if not opt:
                label = ""
                style = None
            else:
                label = str(opt[0])
                style = opt[1] if len(opt) > 1 else None
        else:
            label = str(opt)
            style = None
        normalized.append((label, style))

    if not normalized:
        return None

    def _menu(stdscr):
        curses.curs_set(0)
        curses.start_color()
        try:
            if curses.can_change_color():
                curses.init_color(10, 20, 976, 0)
                curses.init_pair(1, 10, curses.COLOR_BLACK)
            else:
                curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        except curses.error:
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        colors = {
            None: curses.color_pair(1),
            "default": curses.color_pair(1),
            "warning": curses.color_pair(2),
        }
        idx = 0
        offset = 0
        header_lines = header.split("\n") if header else []
        header_height = len(header_lines)
        count = len(normalized)
        while True:
            stdscr.erase()
            rows, cols = stdscr.getmaxyx()
            # draw header
            for i, line in enumerate(header_lines):
                x = max((cols - len(line)) // 2, 0) if center else 0
                stdscr.addstr(i, x, line, curses.color_pair(1))
            view_height = max(1, rows - header_height - 1)
            visible = normalized[offset : offset + view_height]
            for i, (label, style) in enumerate(visible):
                actual = offset + i
                marker = "> " if actual == idx else "  "
                text = marker + label
                x = max((cols - len(text)) // 2, 0) if center else 0
                color = colors.get(style, curses.color_pair(1))
                attr = color | (curses.A_BOLD if actual == idx else curses.A_NORMAL)
                try:
                    stdscr.addstr(header_height + i, x, text, attr)
                except curses.error:
                    pass
            key = stdscr.getch()
            if key == curses.KEY_UP:
                idx = (idx - 1) % count
            elif key == curses.KEY_DOWN:
                idx = (idx + 1) % count
            elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
                return idx
            elif key == 27:
                return None
            if idx < offset:
                offset = idx
            elif idx >= offset + view_height:
                offset = idx - view_height + 1

    return curses.wrapper(_menu)
