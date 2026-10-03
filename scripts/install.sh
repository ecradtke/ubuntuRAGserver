#!/usr/bin/env bash
# ==============================================================================
# K13 AI Stack - installer for Ubuntu Server (native services, no containers)
#
# Usage: ./scripts/install.sh [--units-only] [--upgrade-qdrant] [--skip-qdrant]
#   (no flags)        full install: packages, folders, Python envs, Qdrant, systemd units, start services
#   --units-only      only (re)install the systemd units and the model-switch helper
#   --upgrade-qdrant  download the latest Qdrant even if one is installed
#   --skip-qdrant     do not install Qdrant (you provide your own on port 6333)
# Safe to run repeatedly. Run as your normal user; it uses sudo where needed.
# ==============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

UNITS_ONLY=false; UPGRADE_QDRANT=false; SKIP_QDRANT=false
for a in "$@"; do
  case "$a" in
    --units-only) UNITS_ONLY=true ;; --upgrade-qdrant) UPGRADE_QDRANT=true ;; --skip-qdrant) SKIP_QDRANT=true ;;
    -h|--help) sed -n 2,11p "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "Unknown option: $a" ;;
  esac
done
[[ $EUID -ne 0 ]] || die "Run this as your normal user, not root (it calls sudo when needed)."
command -v sudo >/dev/null || die "sudo is required."
USER_NAME="$(id -un)"
[[ "$BASE_DIR" != *" "* ]] || die "The project path must not contain spaces: $BASE_DIR"

# ---- 1. .env ------------------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
  cp "$BASE_DIR/.env.example" "$ENV_FILE"; chmod 600 "$ENV_FILE"
  ok "Created $ENV_FILE from .env.example"
  echo "    Edit it now (nano $ENV_FILE): FreshRSS + PostgreSQL logins, llama-server path, HF_TOKEN."
  echo "    Then run ./scripts/install.sh again."
  exit 0
fi
chmod 600 "$ENV_FILE"
load_k13_env
STORAGE="$(env_get QDRANT_STORAGE_DIR "$K13_DATA_DIR/qdrant/storage")"
SNAPSHOTS="$(env_get QDRANT_SNAPSHOTS_DIR "$K13_DATA_DIR/qdrant/snapshots")"
grep -qE '^(FRESHRSS_USER|PG_PASS|FRESHRSS_API_PASS)=.*your_' "$ENV_FILE" && \
  warn ".env still has placeholder FreshRSS/PostgreSQL values: the RSS sync will fail until you set them."

install_units() {
  log "Installing systemd units and the model-switch helper..."
  local f name tmp
  for f in "$BASE_DIR"/systemd/*; do
    name="$(basename "$f")"
    sed -e "s#@USER@#${USER_NAME}#g" -e "s#@REPO@#${BASE_DIR}#g" "$f" | sudo tee "/etc/systemd/system/$name" >/dev/null
  done
  tmp="$(mktemp)"
  sed -e "s#@DATA_DIR@#${K13_DATA_DIR}#g" -e "s#@MODELS_DIR@#${MODELS_DIR}#g" "$BASE_DIR/scripts/llm-switch.sh" > "$tmp"
  sudo install -m 755 -o root -g root "$tmp" /usr/local/sbin/k13-llm-switch; rm -f "$tmp"
  echo "${USER_NAME} ALL=(root) NOPASSWD: /usr/local/sbin/k13-llm-switch" | sudo tee /etc/sudoers.d/k13-llm-switch >/dev/null
  sudo chmod 440 /etc/sudoers.d/k13-llm-switch
  if ! sudo visudo -cf /etc/sudoers.d/k13-llm-switch >/dev/null; then
    sudo rm -f /etc/sudoers.d/k13-llm-switch; die "sudoers rule was invalid and has been removed."
  fi
  sudo systemctl daemon-reload
  ok "Units installed."
}

if $UNITS_ONLY; then install_units; exit 0; fi

# ---- 2. OS packages & groups ----------------------------------------------------
log "Installing OS packages..."
if ! sudo apt-get update -qq; then
  warn "apt update reported errors (usually a broken third-party repository in /etc/apt/sources.list.d/)."
  warn "Continuing with the package lists apt already has. Fix or disable the broken repo when you can: sudo apt update"
fi
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git curl ca-certificates python3 python3-venv python3-dev python3-pip \
  build-essential libpq-dev postgresql-client unzip jq util-linux >/dev/null
sudo usermod -aG render,video "$USER_NAME" || true     # NPU/GPU device access for your own shells

# ---- 3. Folders ---------------------------------------------------------------------
log "Creating folders..."
for d in "$MODELS_DIR" "$OV_MODELS_DIR" "$STORAGE" "$SNAPSHOTS" "$K13_DATA_DIR" /opt/k13/qdrant "$BASE_DIR/logs"; do
  sudo mkdir -p "$d"
done
sudo chown -R "$USER_NAME:$USER_NAME" "$K13_DATA_DIR" "$MODELS_DIR" "$OV_MODELS_DIR" "$STORAGE" "$SNAPSHOTS" /opt/k13 "$BASE_DIR/logs"

# ---- 4. Python environments -----------------------------------------------------------
log "Creating the web/RSS Python environment (.venv)..."
python3 -m venv "$BASE_DIR/.venv"
"$BASE_DIR/.venv/bin/pip" install -q --upgrade pip
"$BASE_DIR/.venv/bin/pip" install -q -r "$BASE_DIR/web/requirements.txt" || {
  warn "pip install failed; retrying without the binary psycopg2 wheel..."
  grep -v psycopg2 "$BASE_DIR/web/requirements.txt" | "$BASE_DIR/.venv/bin/pip" install -q -r /dev/stdin
  "$BASE_DIR/.venv/bin/pip" install -q psycopg2 || die "Could not install psycopg2 (needs libpq-dev, installed above)."
}

OV_PY="$(env_get OPENVINO_PYTHON)"
if [[ -n "$OV_PY" ]]; then
  log "Using your OpenVINO Python: $OV_PY"
else
  OV_PY="$BASE_DIR/.venv-openvino/bin/python"
  if [[ ! -x "$OV_PY" ]]; then
    log "Creating .venv-openvino (reuses system-installed packages such as your existing OpenVINO)..."
    python3 -m venv --system-site-packages "$BASE_DIR/.venv-openvino"
  fi
  "$BASE_DIR/.venv-openvino/bin/pip" install -q --upgrade pip
  "$BASE_DIR/.venv-openvino/bin/pip" install -q -r "$BASE_DIR/embed/requirements.txt" \
    || warn "pip could not install every embedding package (no wheels for this Python?). Set OPENVINO_PYTHON in .env to the Python that already has OpenVINO, then rerun."
fi
missing="$("$OV_PY" - <<'PY' 2>/dev/null || true
import importlib
miss=[m for m in ("openvino","fastapi","uvicorn","transformers","numpy") if importlib.util.find_spec(m) is None]
print(" ".join(miss))
PY
)"
[[ -z "$missing" ]] && ok "OpenVINO Python has everything the embedding server needs." \
  || warn "OpenVINO Python is missing: $missing  -> $OV_PY -m pip install -r $BASE_DIR/embed/requirements.txt"

# ---- 5. Qdrant (native binary) ----------------------------------------------------------
install_qdrant() {
  [[ "$(uname -m)" == x86_64 ]] || die "Qdrant auto-install supports x86_64 only (found $(uname -m))."
  local tmp urls url ui
  tmp="$(mktemp -d)"
  log "Finding the latest Qdrant release..."
  urls="$(curl -fsSL https://api.github.com/repos/qdrant/qdrant/releases/latest | jq -r '.assets[].browser_download_url' | grep -E 'x86_64-unknown-linux-(musl|gnu)\.tar\.gz$' || true)"
  url="${QDRANT_URL_OVERRIDE:-$(grep musl <<<"$urls" | head -n1)}"
  [[ -n "$url" ]] || url="$(grep gnu <<<"$urls" | head -n1)"
  [[ -n "$url" ]] || die "Could not find a Qdrant release asset. Download it manually from github.com/qdrant/qdrant/releases and set QDRANT_URL_OVERRIDE=<tar.gz url>."
  log "Downloading $url"
  curl -fL --retry 3 -o "$tmp/qdrant.tgz" "$url"
  tar -xzf "$tmp/qdrant.tgz" -C "$tmp"
  install -m 755 "$(find "$tmp" -type f -name qdrant | head -n1)" /opt/k13/qdrant/qdrant
  ui="$(curl -fsSL https://api.github.com/repos/qdrant/qdrant-web-ui/releases/latest | jq -r '.assets[].browser_download_url' | grep -E 'dist-qdrant\.zip$' | head -n1 || true)"
  if [[ -n "$ui" ]] && curl -fL --retry 3 -o "$tmp/ui.zip" "$ui" && unzip -q -o "$tmp/ui.zip" -d "$tmp/ui"; then
    rm -rf /opt/k13/qdrant/static && mkdir -p /opt/k13/qdrant/static
    cp -r "$(find "$tmp/ui" -maxdepth 1 -mindepth 1 -type d -name dist | head -n1 || echo "$tmp/ui")"/. /opt/k13/qdrant/static/
    ok "Qdrant dashboard installed."
  else
    warn "Could not install the Qdrant web dashboard (the database itself works without it)."
  fi
  rm -rf "$tmp"
  ok "Qdrant installed: $(/opt/k13/qdrant/qdrant --version 2>/dev/null | head -n1)"
}
if $SKIP_QDRANT; then log "Skipping Qdrant install."
elif $UPGRADE_QDRANT || [[ ! -x /opt/k13/qdrant/qdrant ]]; then install_qdrant
else ok "Qdrant already installed (use --upgrade-qdrant to update)."; fi

# ---- 6. llama.cpp check --------------------------------------------------------------------
LLAMA_BIN="$(env_get LLAMA_SERVER_BIN /usr/local/bin/llama-server)"
if [[ ! -x "$LLAMA_BIN" ]] && ! command -v llama-server >/dev/null; then
  warn "llama-server not found at $LLAMA_BIN. Set LLAMA_SERVER_BIN in .env (find it with: command -v llama-server)."
fi

# ---- 7. systemd ----------------------------------------------------------------------------------
install_units
sudo systemctl enable k13-qdrant.service k13-embed.service k13-web.service k13-rss-sync.timer k13-llama.service >/dev/null 2>&1
sudo systemctl restart k13-qdrant.service k13-embed.service
sudo systemctl restart k13-web.service
sudo systemctl start k13-rss-sync.timer
HAVE_MODEL=false
if [[ -s "$K13_DATA_DIR/active_model" ]] || [[ -n "$(env_get LLAMA_MODEL)" ]]; then HAVE_MODEL=true; fi
if $HAVE_MODEL; then sudo systemctl restart k13-llama.service; fi

# ---- 8. Verify ---------------------------------------------------------------------------------------
log "Waiting for services (the embedding model can take a few minutes on its first start)..."
for i in $(seq 1 30); do
  q=$(curl -fs --max-time 3 "http://127.0.0.1:${QDRANT_PORT}/healthz" >/dev/null 2>&1 && echo up || echo down)
  e=$(curl -fs --max-time 3 "http://${EMBED_HOST}:${EMBED_PORT}/healthz" >/dev/null 2>&1 && echo up || echo down)
  [[ "$q" == up && "$e" == up ]] && break
  echo "    qdrant: $q  embed: $e  ($i/30)"; sleep 4
done
echo
ok "Install finished. Next steps:"
echo "  - Status:        ./scripts/k13ctl.sh status"
$HAVE_MODEL || echo "  - Get a model:   ./scripts/download-model.sh REPO FILE.gguf --switch    (then the chat LLM starts)"
echo "  - Web portal:    http://$(hostname -I | awk '{print $1}'):$(env_get WEB_PORT 8501)"
echo "  - Backups:       ./scripts/install-cron.sh"
echo "  (Log out and back in once so your shell picks up the render/video groups.)"
