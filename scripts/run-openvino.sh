#!/usr/bin/env bash
# systemd ExecStart for k13-openvino: OpenVINO LLM (GPU) + embeddings (NPU) server
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
load_k13_env
PY="$(env_get OPENVINO_PYTHON)"; PY="${PY:-$BASE_DIR/.venv-openvino/bin/python}"
[[ -x "$PY" ]] || die "OpenVINO python not found: $PY (run scripts/install.sh or set OPENVINO_PYTHON in .env)"
SETUP="$(env_get OPENVINO_SETUPVARS)"
if [[ -n "$SETUP" ]]; then [[ -f "$SETUP" ]] || die "OPENVINO_SETUPVARS not found: $SETUP"; set +u; source "$SETUP"; set -u; fi

export DEFAULT_MODEL="$(env_get DEFAULT_MODEL OpenVINO/Qwen3-8B-int4-ov)"
export EMBED_MODEL="$(env_get EMBED_MODEL BAAI/bge-base-en-v1.5)"
export LLM_DEVICE="$(env_get LLM_DEVICE GPU)" EMBED_DEVICE="$(env_get EMBED_DEVICE NPU)"
export LLM_THINKING="$(env_get LLM_THINKING off)" MAX_NEW_TOKENS="$(env_get MAX_NEW_TOKENS 2048)"
export SWITCH_ALLOW_CPU="$(env_get SWITCH_ALLOW_CPU off)"
export HF_TOKEN="$(env_get HF_TOKEN)"
export MODELS_DIR="$OV_MODELS_DIR"
export SERVER_HOST="$OV_HOST" SERVER_PORT="$OV_PORT"
export ZE_INIT_ALL_DRIVER_TYPES_ENABLE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
mkdir -p "$OV_MODELS_DIR"
exec "$PY" "$BASE_DIR/openvino/server.py"
