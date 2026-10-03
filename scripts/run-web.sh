#!/usr/bin/env bash
# systemd ExecStart for k13-web: Streamlit portal
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
load_k13_env
PY="$BASE_DIR/.venv/bin/python"
[[ -x "$PY" ]] || die "Project venv missing (run scripts/install.sh)"
export QDRANT_URL="$(env_get QDRANT_URL "http://127.0.0.1:${QDRANT_PORT}")"
export LLAMA_URL="$(env_get LLAMA_URL "http://${LLAMA_HOST}:${LLAMA_PORT}")"
export EMBED_URL="$(env_get EMBED_URL "http://${EMBED_HOST}:${EMBED_PORT}")"
export QDRANT_COLLECTION="$(env_get QDRANT_COLLECTION rss_articles)"
export LLM_THINKING="$(env_get LLM_THINKING off)" MAX_NEW_TOKENS="$(env_get MAX_NEW_TOKENS 2048)"
export LLM_SWITCH_BIN="${LLM_SWITCH_BIN:-/usr/local/sbin/k13-llm-switch}"
cd "$BASE_DIR/web"
exec "$PY" -m streamlit run app.py --server.port "$(env_get WEB_PORT 8501)" --server.address "$(env_get WEB_HOST 0.0.0.0)" \
     --server.headless true --browser.gatherUsageStats false
