#!/usr/bin/env bash
# Download a GGUF model from Hugging Face into MODELS_DIR (resumable).
#   ./scripts/download-model.sh --preset qwen3-8b --switch
#   ./scripts/download-model.sh REPO FILE.gguf [--switch]
#   ./scripts/download-model.sh --list-presets
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
load_k13_env

# Presets: the models from the previous OpenVINO project, as llama.cpp GGUF files (Q4_K_M = best size/quality balance).
# Repo and file names are the usual ones on huggingface.co; if one 404s, fix it here or pass REPO FILE directly.
declare -A PRESETS=(
  [qwen3-8b]="Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf          ~5 GB    best quality that fits (previous default)"
  [qwen3-4b]="Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf          ~2.5 GB  fast, good all-rounder"
  [qwen3-14b]="Qwen/Qwen3-14B-GGUF Qwen3-14B-Q4_K_M.gguf       ~9 GB    bigger; needs GPU/CPU layer split on this machine"
  [phi-4-mini]="unsloth/Phi-4-mini-instruct-GGUF Phi-4-mini-instruct-Q4_K_M.gguf   ~2.5 GB  no hidden reasoning"
  [phi-3.5-mini]="bartowski/Phi-3.5-mini-instruct-GGUF Phi-3.5-mini-instruct-Q4_K_M.gguf   ~2.3 GB  small and quick"
  [llama-3.2-3b]="bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf   ~2 GB    smallest, reliable"
  [gemma-3-4b]="unsloth/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf   ~2.5 GB  good writing style"
)

usage() {
  echo "Usage:"
  echo "  ./scripts/download-model.sh --preset NAME [--switch]     (see --list-presets)"
  echo "  ./scripts/download-model.sh REPO FILE.gguf [--switch]    (REPO and FILE are the Hugging Face repo and file names)"
  echo "  ./scripts/download-model.sh --list-presets"
}

if [[ "${1:-}" == "--list-presets" ]]; then
  for k in "${!PRESETS[@]}"; do printf '  %-14s %s\n' "$k" "${PRESETS[$k]#* * }"; done | sort
  exit 0
fi

if [[ "${1:-}" == "--preset" ]]; then
  [[ -n "${2:-}" && -n "${PRESETS[$2]:-}" ]] || { usage; die "unknown preset '${2:-}' (try --list-presets)"; }
  read -r REPO FILE _ <<<"${PRESETS[$2]}"; SWITCH="${3:-}"
else
  REPO="${1:-}"; FILE="${2:-}"; SWITCH="${3:-}"
  [[ -n "$REPO" && -n "$FILE" ]] || { usage; exit 1; }
fi
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
