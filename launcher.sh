#!/usr/bin/env bash
set -e

GREEN="\033[32m"
RESET="\033[0m"
BOLD="\033[1m"

clear
tput civis

ROWS=$(tput lines)
COLS=$(tput cols)

HEADER_HEIGHT=6
HEADER_TOP=0
HEADER_BOTTOM=$((HEADER_TOP + HEADER_HEIGHT - 1))

BOX_WIDTH=34
BOX_HEIGHT=4
BOX_TOP=$((HEADER_BOTTOM + 2))
BOX_LEFT=$(((COLS - BOX_WIDTH) / 2))
BOX_RIGHT=$((BOX_LEFT + BOX_WIDTH - 1))
BOX_BOTTOM=$((BOX_TOP + BOX_HEIGHT - 1))
PROMPT_ROW=$((BOX_BOTTOM + 1))

draw_header() {
  local start_col=$(((COLS - 68) / 2))
  tput cup $HEADER_TOP $start_col
  printf "${BOLD}${GREEN}████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗\n"
  tput cup $((HEADER_TOP+1)) $start_col
  printf "╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║\n"
  tput cup $((HEADER_TOP+2)) $start_col
  printf "   ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║\n"
  tput cup $((HEADER_TOP+3)) $start_col
  printf "   ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║\n"
  tput cup $((HEADER_TOP+4)) $start_col
  printf "   ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗\n"
  tput cup $((HEADER_TOP+5)) $start_col
  printf "   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝${RESET}"
}

draw_box() {
  tput cup $BOX_TOP $BOX_LEFT
  printf "${GREEN}┌"
  printf '─%.0s' $(seq 1 $((BOX_WIDTH-2)))
  printf "┐"
  for ((i=1; i<=BOX_HEIGHT-2; i++)); do
    tput cup $((BOX_TOP+i)) $BOX_LEFT
    printf "│"
    tput cup $((BOX_TOP+i)) $((BOX_LEFT+BOX_WIDTH-1))
    printf "│"
  done
  tput cup $BOX_BOTTOM $BOX_LEFT
  printf "└"
  printf '─%.0s' $(seq 1 $((BOX_WIDTH-2)))
  printf "┘${RESET}"
}

print_options() {
  tput cup $((BOX_TOP+1)) $((BOX_LEFT+2))
  printf "${GREEN}1) Start TerminalAI"
  tput cup $((BOX_TOP+2)) $((BOX_LEFT+2))
  printf "2) Scan Shodan${RESET}"
}

draw_header
draw_box
print_options

python rain.py --header-top "$HEADER_TOP" --header-bottom "$HEADER_BOTTOM" \
  --box-top "$BOX_TOP" --box-bottom "$BOX_BOTTOM" --box-left "$BOX_LEFT" \
  --box-right "$BOX_RIGHT" --prompt-row "$PROMPT_ROW" &
R_PID=$!

cleanup() {
  kill $R_PID 2>/dev/null || true
  tput cnorm
  clear
}
trap cleanup EXIT

tput cup $PROMPT_ROW $BOX_LEFT
read -p "Select option: " choice

case "$choice" in
  1) cleanup; python TerminalAI.py "$@" ;;
  2) cleanup; python shodanscan.py "$@" ;;
  *) cleanup; echo "Invalid selection" >&2; exit 1 ;;
esac
