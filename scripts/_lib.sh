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
  OV_MODELS_DIR="$(env_get OV_MODELS_DIR "$K13_DATA_DIR/openvino")"
  OV_HOST="$(env_get OV_HOST 127.0.0.1)"; OV_PORT="$(env_get OV_PORT 11434)"
  QDRANT_PORT="$(env_get QDRANT_PORT 6333)"
  export K13_DATA_DIR OV_MODELS_DIR OV_HOST OV_PORT QDRANT_PORT
}

log()  { printf '[*] %s\n' "$*"; }
ok()   { printf '[+] %s\n' "$*"; }
warn() { printf '[!] %s\n' "$*" >&2; }
die()  { printf '[!] ERROR: %s\n' "$*" >&2; exit 1; }
