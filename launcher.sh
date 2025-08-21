#!/usr/bin/env bash
set -e

GREEN="\033[32m"
RESET="\033[0m"
BOLD="\033[1m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

clear
tput civis

ROWS=$(tput lines)
COLS=$(tput cols)

HEADER_HEIGHT=6
HEADER_TOP=0
HEADER_BOTTOM=$((HEADER_TOP + HEADER_HEIGHT - 1))
HEADER_WIDTH=76
HEADER_LEFT=$(((COLS - HEADER_WIDTH) / 2))
HEADER_RIGHT=$((HEADER_LEFT + HEADER_WIDTH - 1))

BOX_WIDTH=34
BOX_HEIGHT=4
BOX_TOP=$((HEADER_BOTTOM + 2))
BOX_LEFT=$(((COLS - BOX_WIDTH) / 2))
BOX_RIGHT=$((BOX_LEFT + BOX_WIDTH - 1))
BOX_BOTTOM=$((BOX_TOP + BOX_HEIGHT - 1))

# convert to 1-based coordinates for rain exclusion
RAIN_HEADER_TOP=$((HEADER_TOP+1))
RAIN_HEADER_BOTTOM=$((HEADER_BOTTOM+1))
RAIN_HEADER_LEFT=$((HEADER_LEFT+1))
RAIN_HEADER_RIGHT=$((HEADER_RIGHT+1))
RAIN_BOX_TOP=$((BOX_TOP+1))
RAIN_BOX_BOTTOM=$((BOX_BOTTOM+1))
RAIN_BOX_LEFT=$((BOX_LEFT+1))
RAIN_BOX_RIGHT=$((BOX_RIGHT+1))

draw_header() {
  tput cup $HEADER_TOP $HEADER_LEFT
  printf "${BOLD}${GREEN}████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗      █████╗ ██╗\n"
  tput cup $((HEADER_TOP+1)) $HEADER_LEFT
  printf "╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║     ██╔══██╗██║\n"
  tput cup $((HEADER_TOP+2)) $HEADER_LEFT
  printf "   ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║     ███████║██║\n"
  tput cup $((HEADER_TOP+3)) $HEADER_LEFT
  printf "   ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║     ██╔══██║██║\n"
  tput cup $((HEADER_TOP+4)) $HEADER_LEFT
  printf "   ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗██║  ██║██║\n"
  tput cup $((HEADER_TOP+5)) $HEADER_LEFT
  printf "   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝${RESET}"
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

  python3 "$SCRIPT_DIR/scripts/rain.py" --persistent --no-clear \
    --exclude "$RAIN_HEADER_TOP,$RAIN_HEADER_BOTTOM,$RAIN_HEADER_LEFT,$RAIN_HEADER_RIGHT" \
    --exclude "$RAIN_BOX_TOP,$RAIN_BOX_BOTTOM,$RAIN_BOX_LEFT,$RAIN_BOX_RIGHT" &
R_PID=$!

draw_header
draw_box
print_options

cleanup() {
  kill $R_PID 2>/dev/null || true
  tput cnorm
  clear
}
trap cleanup EXIT
while true; do
  read -rsn1 choice
  case "$choice" in
    1) cleanup; python3 "$SCRIPT_DIR/scripts/TerminalAI.py" "$@"; break ;;
    2) cleanup; python3 "$SCRIPT_DIR/scripts/shodanscan.py" "$@"; break ;;
  esac
done
