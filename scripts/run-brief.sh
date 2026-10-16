#!/usr/bin/env bash
# Run the daily news brief. Dry run unless you pass --post.   Usage: scripts/run-brief.sh [brief.py args]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
load_k13_env
PY="$BASE_DIR/.venv/bin/python"
[[ -x "$PY" ]] || die "Project venv missing (run scripts/install.sh)"
exec "$PY" "$BASE_DIR/brief/brief.py" "$@"
