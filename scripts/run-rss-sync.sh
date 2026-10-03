#!/usr/bin/env bash
# systemd ExecStart for k13-rss-sync (run every 5 minutes by k13-rss-sync.timer)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
load_k13_env
PY="$BASE_DIR/.venv/bin/python"
[[ -x "$PY" ]] || die "Project venv missing (run scripts/install.sh)"
for k in FRESHRSS_BASE_URL FRESHRSS_USER FRESHRSS_API_PASS PG_HOST PG_PORT PG_DB PG_USER PG_PASS; do
  export "$k=$(env_get "$k")"
done
export EMBEDDING_HOST="http://127.0.0.1:${OV_PORT}" OLLAMA_HOST="http://127.0.0.1:${OV_PORT}"
export EMBED_MODEL=bge-base-en-v1.5
export QDRANT_HOST=127.0.0.1 QDRANT_PORT="$QDRANT_PORT"
export PYTHONUNBUFFERED=1
exec "$PY" "$BASE_DIR/rss/sync_freshrss.py"
