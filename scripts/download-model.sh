#!/usr/bin/env bash
# Download a GGUF model from Hugging Face into MODELS_DIR (resumable).
#   ./scripts/download-model.sh <hf-repo> <file.gguf> [--switch]
# Example (check the exact names on huggingface.co first):
#   ./scripts/download-model.sh unsloth/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --switch
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
load_k13_env
REPO="${1:-}"; FILE="${2:-}"; SWITCH="${3:-}"
[[ -n "$REPO" && -n "$FILE" ]] || { sed -n 2,6p "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }
[[ "$FILE" =~ ^[A-Za-z0-9._+-]+\.gguf$ ]] || die "file must be a plain .gguf file name (no folders)"
mkdir -p "$MODELS_DIR"
log "Free space in $MODELS_DIR: $(df -h --output=avail "$MODELS_DIR" | tail -1 | tr -d ' ')"
HF_TOKEN="$(env_get HF_TOKEN)"
AUTH=(); [[ -n "$HF_TOKEN" ]] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"
log "Downloading $URL"
curl -fL --retry 3 -C - "${AUTH[@]}" -o "$MODELS_DIR/$FILE.part" "$URL" || die "download failed (check the repo and file names, or set HF_TOKEN for gated models)"
mv -f "$MODELS_DIR/$FILE.part" "$MODELS_DIR/$FILE"
ok "Saved $MODELS_DIR/$FILE ($(du -h "$MODELS_DIR/$FILE" | cut -f1))"
if [[ "$SWITCH" == "--switch" ]]; then sudo -n /usr/local/sbin/k13-llm-switch "$FILE" || sudo /usr/local/sbin/k13-llm-switch "$FILE"; fi
