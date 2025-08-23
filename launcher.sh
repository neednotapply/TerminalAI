#!/usr/bin/env bash
set -e

# Resolve directory of this script, trimming any Windows carriage returns
SCRIPT_PATH="${BASH_SOURCE[0]%$'\r'}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR%$'\r'}"

python3 "$SCRIPT_DIR/scripts/launcher.py" "$@"
