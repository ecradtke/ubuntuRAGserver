#!/usr/bin/env bash
# k13-llm-switch - choose the GGUF model llama-server runs, then restart it.
# Installed to /usr/local/sbin/k13-llm-switch by install.sh and allowed (only this script) through sudoers,
# so the web portal can switch models without a password. Input is strictly validated.
# Re-run scripts/install.sh after changing K13_DATA_DIR / MODELS_DIR in .env.
#   k13-llm-switch --list          list model files and the active one
#   k13-llm-switch FILE.gguf       make FILE.gguf the active model and restart k13-llama
set -euo pipefail
# Paths are baked in by install.sh (this root-owned copy never reads the user-editable .env).
DATA_DIR="@DATA_DIR@"
MODELS_DIR="@MODELS_DIR@"
ACTIVE_FILE="$DATA_DIR/active_model"

case "${1:-}" in
  --list)
    active="$(cat "$ACTIVE_FILE" 2>/dev/null || true)"
    shopt -s nullglob
    for f in "$MODELS_DIR"/*.gguf; do
      b="$(basename "$f")"; printf '%s\t%s\t%s\n' "$b" "$(stat -c %s "$f")" "$([[ "$b" == "$active" ]] && echo active || echo -)"
    done
    ;;
  *.gguf)
    name="$1"
    [[ "$name" =~ ^[A-Za-z0-9._+-]+\.gguf$ ]] || { echo "invalid model name" >&2; exit 2; }
    [[ "$name" != *..* ]] || { echo "invalid model name" >&2; exit 2; }
    [[ -f "$MODELS_DIR/$name" ]] || { echo "model not found in $MODELS_DIR: $name" >&2; exit 3; }
    tmp="$(mktemp "$DATA_DIR/.active_model.XXXXXX")"
    printf '%s\n' "$name" > "$tmp"
    chmod 644 "$tmp"; mv -f "$tmp" "$ACTIVE_FILE"
    chown "$(stat -c %U "$DATA_DIR")" "$ACTIVE_FILE" 2>/dev/null || true
    systemctl reset-failed k13-llama.service 2>/dev/null || true
    systemctl restart k13-llama.service
    echo "switched to $name"
    ;;
  *)
    echo "usage: k13-llm-switch --list | FILE.gguf" >&2; exit 64 ;;
esac
