# shellcheck shell=bash
# Shared helpers, sourced by the other scripts:  source "$(dirname "$0")/_lib.sh"
# The .env file is PARSED, never sourced, so values with spaces or $ cannot run code.
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(dirname "$_LIB_DIR")}"
ENV_FILE="${ENV_FILE:-${BASE_DIR}/.env}"

env_get() {                 # env_get KEY [default]  -> value from the real environment, then .env, then default
  local v="${!1:-}"
  if [[ -z "$v" && -f "$ENV_FILE" ]]; then
    v="$(grep -E "^[[:space:]]*$1=" "$ENV_FILE" | tail -n1 | cut -d= -f2- || true)"
    v="${v%"${v##*[![:space:]]}"}"; v="${v#"${v%%[![:space:]]*}"}"       # trim whitespace
    v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"                    # strip quotes
    v="${v%$'\r'}"
  fi
  printf '%s' "${v:-${2:-}}"
}

# Load the variables the run-* wrappers need, with defaults, into the environment
load_k13_env() {
  K13_DATA_DIR="$(env_get K13_DATA_DIR /var/lib/k13)"
  MODELS_DIR="$(env_get MODELS_DIR "$K13_DATA_DIR/models")"
  OV_MODELS_DIR="$(env_get OV_MODELS_DIR "$K13_DATA_DIR/openvino")"
  EMBED_HOST="$(env_get EMBED_HOST 127.0.0.1)"; EMBED_PORT="$(env_get EMBED_PORT 11434)"
  LLAMA_HOST="$(env_get LLAMA_HOST 127.0.0.1)"; LLAMA_PORT="$(env_get LLAMA_PORT 8080)"
  QDRANT_PORT="$(env_get QDRANT_PORT 6333)"
  export K13_DATA_DIR MODELS_DIR OV_MODELS_DIR EMBED_HOST EMBED_PORT LLAMA_HOST LLAMA_PORT QDRANT_PORT
}

log()  { printf '[*] %s\n' "$*"; }
ok()   { printf '[+] %s\n' "$*"; }
warn() { printf '[!] %s\n' "$*" >&2; }
die()  { printf '[!] ERROR: %s\n' "$*" >&2; exit 1; }
