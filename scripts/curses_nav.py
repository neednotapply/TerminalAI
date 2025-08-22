import curses
import curses.textpad

def get_input(prompt: str) -> str:
    """Prompt for input with basic line editing using curses."""
    def _inner(stdscr):
        curses.curs_set(1)
        curses.echo()
        stdscr.addstr(prompt)
        stdscr.refresh()
        win = curses.newwin(1, curses.COLS - len(prompt) - 1, 0, len(prompt))
        box = curses.textpad.Textbox(win)
        text = box.edit().strip()
        return text
    return curses.wrapper(_inner)


def interactive_menu(header: str, options: list[str]) -> int | None:
    """Simple vertical menu navigated with arrow keys."""
    def _menu(stdscr):
        curses.curs_set(0)
        idx = 0
        offset = 0
        while True:
            stdscr.erase()
            stdscr.addstr(0, 0, header)
            rows, cols = stdscr.getmaxyx()
            view_height = max(1, rows - 2)
            visible = options[offset:offset + view_height]
            for i, opt in enumerate(visible):
                actual = offset + i
                marker = "> " if actual == idx else "  "
                attr = curses.A_BOLD if actual == idx else curses.A_NORMAL
                stdscr.addstr(i + 1, 0, marker + opt, attr)
            key = stdscr.getch()
            if key == curses.KEY_UP:
                idx = (idx - 1) % len(options)
            elif key == curses.KEY_DOWN:
                idx = (idx + 1) % len(options)
            elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
                return idx
            elif key == 27:
                return None
            if idx < offset:
                offset = idx
            elif idx >= offset + view_height:
                offset = idx - view_height + 1
    return curses.wrapper(_menu)
