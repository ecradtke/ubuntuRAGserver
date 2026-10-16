#!/usr/bin/env bash
# systemd ExecStart for k13-web: Streamlit portal
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
load_k13_env
PY="$BASE_DIR/.venv/bin/python"
[[ -x "$PY" ]] || die "Project venv missing (run scripts/install.sh)"
export QDRANT_URL="$(env_get QDRANT_URL "http://127.0.0.1:${QDRANT_PORT}")"
export OLLAMA_HOST="$(env_get OPENVINO_URL "http://127.0.0.1:${OV_PORT}")"
export QDRANT_COLLECTION="$(env_get QDRANT_COLLECTION rss_articles)"
export DEFAULT_MODEL="$(env_get DEFAULT_MODEL OpenVINO/Qwen3-8B-int4-ov)"
export TOP_K_MAX="$(env_get TOP_K_MAX 100)" RAG_CONTEXT_TOKEN_BUDGET="$(env_get RAG_CONTEXT_TOKEN_BUDGET 8000)"
export QUERY_STATS_LOG="$(env_get QUERY_STATS_LOG "${K13_DATA_DIR:-/var/lib/k13}/query_stats.jsonl")"
cd "$BASE_DIR/web"
exec "$PY" -m streamlit run app.py --server.port "$(env_get WEB_PORT 8501)" --server.address "$(env_get WEB_HOST 0.0.0.0)" \
     --server.headless true --browser.gatherUsageStats false
