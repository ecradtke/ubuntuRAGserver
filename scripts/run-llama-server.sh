#!/usr/bin/env bash
# systemd ExecStart for k13-llama: llama.cpp llama-server (OpenAI-compatible chat API)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
load_k13_env
BIN="$(env_get LLAMA_SERVER_BIN /usr/local/bin/llama-server)"
if [[ ! -x "$BIN" ]]; then BIN="$(command -v llama-server || true)"; fi
[[ -n "$BIN" && -x "$BIN" ]] || { warn "llama-server not found. Set LLAMA_SERVER_BIN in .env"; exit 78; }

# Active model: the file written by the portal's model switcher wins over LLAMA_MODEL from .env
MODEL="$(cat "$K13_DATA_DIR/active_model" 2>/dev/null || true)"
MODEL="${MODEL:-$(env_get LLAMA_MODEL)}"
if [[ -z "$MODEL" ]]; then
  warn "No model selected. Download one (scripts/download-model.sh) and switch to it (portal sidebar or: sudo k13-llm-switch FILE.gguf)."
  exit 78                                    # 78 = configuration error: systemd will not restart-loop
fi
[[ -f "$MODELS_DIR/$MODEL" ]] || { warn "Model file missing: $MODELS_DIR/$MODEL"; exit 78; }

PRE="$(env_get LLAMA_PRE_SCRIPT)"
if [[ -n "$PRE" ]]; then [[ -f "$PRE" ]] || die "LLAMA_PRE_SCRIPT not found: $PRE"; set +u; source "$PRE"; set -u; fi

ARGS=(--model "$MODELS_DIR/$MODEL" --alias "${MODEL%.gguf}"
      --host "$LLAMA_HOST" --port "$LLAMA_PORT"
      --ctx-size "$(env_get LLAMA_CTX 16384)" --n-gpu-layers "$(env_get LLAMA_GPU_LAYERS 99)" --jinja)
EXTRA="$(env_get LLAMA_EXTRA_ARGS)"
# shellcheck disable=SC2206
[[ -n "$EXTRA" ]] && ARGS+=($EXTRA)
log "Starting llama-server: $MODEL on ${LLAMA_HOST}:${LLAMA_PORT}"
exec "$BIN" "${ARGS[@]}"
