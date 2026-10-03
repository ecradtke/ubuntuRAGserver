#!/usr/bin/env bash
# systemd ExecStart for k13-embed: OpenVINO embedding server (Intel NPU, CPU fallback)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
load_k13_env
PY="$(env_get OPENVINO_PYTHON)"; PY="${PY:-$BASE_DIR/.venv-openvino/bin/python}"
[[ -x "$PY" ]] || die "OpenVINO python not found: $PY (run scripts/install.sh or set OPENVINO_PYTHON in .env)"
SETUP="$(env_get OPENVINO_SETUPVARS)"
if [[ -n "$SETUP" ]]; then [[ -f "$SETUP" ]] || die "OPENVINO_SETUPVARS not found: $SETUP"; set +u; source "$SETUP"; set -u; fi
export EMBED_MODEL="$(env_get EMBED_MODEL BAAI/bge-small-en-v1.5)"
export EMBED_DEVICE="$(env_get EMBED_DEVICE NPU)"
export HF_TOKEN="$(env_get HF_TOKEN)"
export OV_MODELS_DIR
export ZE_INIT_ALL_DRIVER_TYPES_ENABLE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
mkdir -p "$OV_MODELS_DIR"
exec "$PY" "$BASE_DIR/embed/embed_server.py"
