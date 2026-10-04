#!/usr/bin/env bash
# ==============================================================================
# K13 AI Stack - installer for Ubuntu Server (native services, no containers)
#
# Usage: ./scripts/install.sh [--units-only] [--upgrade-qdrant] [--skip-qdrant] [--keep-embed-model] [--with-share] [--with-backups]
#   (no flags)        full install: packages, folders, Python envs, Qdrant, systemd units, start services
#   --units-only      only (re)install the systemd units
#   --upgrade-qdrant  download the latest Qdrant even if one is installed
#   --skip-qdrant     do not install Qdrant (you provide your own on port 6333)
#   --with-share      also create the dropit folder and share it on the network (Samba; asks for a share password)
#   --with-backups    also schedule the nightly backups (cron, 04:00 project+Qdrant, 04:15 PostgreSQL)
#   --keep-embed-model  do not reset EMBED_MODEL in .env to BGE Base (BAAI/bge-base-en-v1.5)
# Safe to run repeatedly. Run as your normal user; it uses sudo where needed.
# ==============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

UNITS_ONLY=false; UPGRADE_QDRANT=false; SKIP_QDRANT=false; KEEP_EMBED=false; WITH_SHARE=false; WITH_BACKUPS=false
for a in "$@"; do
  case "$a" in
    --units-only) UNITS_ONLY=true ;; --upgrade-qdrant) UPGRADE_QDRANT=true ;; --skip-qdrant) SKIP_QDRANT=true ;; --keep-embed-model) KEEP_EMBED=true ;; --with-share) WITH_SHARE=true ;; --with-backups) WITH_BACKUPS=true ;;
    -h|--help) sed -n 2,11p "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "Unknown option: $a" ;;
  esac
done
[[ $EUID -ne 0 ]] || die "Run this as your normal user, not root (it calls sudo when needed)."
command -v sudo >/dev/null || die "sudo is required."
USER_NAME="$(id -un)"
chmod +x "$BASE_DIR"/scripts/*.sh 2>/dev/null || true     # git on some systems drops the executable bit
[[ "$BASE_DIR" != *" "* ]] || die "The project path must not contain spaces: $BASE_DIR"

# ---- 1. .env ------------------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
  cp "$BASE_DIR/.env.example" "$ENV_FILE"; chmod 600 "$ENV_FILE"
  ok "Created $ENV_FILE from .env.example"
  echo "    Edit it now (nano $ENV_FILE): FreshRSS + PostgreSQL logins, HF_TOKEN, and OPENVINO_PYTHON if your OpenVINO lives elsewhere."
  echo "    Then run ./scripts/install.sh again."
  exit 0
fi
chmod 600 "$ENV_FILE"
# The embedding model defaults to BGE Base (768 dims) on every install; --keep-embed-model opts out.
if ! $KEEP_EMBED && ! $UNITS_ONLY; then
  cur="$(env_get EMBED_MODEL)"
  if [[ "$cur" != "BAAI/bge-base-en-v1.5" ]]; then
    if grep -qE '^[[:space:]]*EMBED_MODEL=' "$ENV_FILE"; then
      sed -i -E 's#^[[:space:]]*EMBED_MODEL=.*#EMBED_MODEL=BAAI/bge-base-en-v1.5#' "$ENV_FILE"
    else
      printf '\nEMBED_MODEL=BAAI/bge-base-en-v1.5\n' >> "$ENV_FILE"
    fi
    ok "EMBED_MODEL set to BAAI/bge-base-en-v1.5 in .env (was: ${cur:-unset}). The RSS sync re-indexes Qdrant if the vector size changed."
  fi
fi
load_k13_env
STORAGE="$(env_get QDRANT_STORAGE_DIR "$K13_DATA_DIR/qdrant/storage")"
SNAPSHOTS="$(env_get QDRANT_SNAPSHOTS_DIR "$K13_DATA_DIR/qdrant/snapshots")"
grep -qE '^(FRESHRSS_USER|PG_PASS|FRESHRSS_API_PASS)=.*your_' "$ENV_FILE" && \
  warn ".env still has placeholder FreshRSS/PostgreSQL values: the RSS sync will fail until you set them."

install_units() {
  log "Installing systemd units..."
  local f name
  for f in "$BASE_DIR"/systemd/*; do
    name="$(basename "$f")"
    sed -e "s#@USER@#${USER_NAME}#g" -e "s#@REPO@#${BASE_DIR}#g" "$f" | sudo tee "/etc/systemd/system/$name" >/dev/null
  done
  # clean up units/helpers from the earlier llama.cpp-based layout, if they were ever installed
  for old in k13-llama.service k13-embed.service; do
    if [[ -f "/etc/systemd/system/$old" ]]; then sudo systemctl disable --now "$old" 2>/dev/null || true; sudo rm -f "/etc/systemd/system/$old"; fi
  done
  sudo rm -f /etc/sudoers.d/k13-llm-switch /usr/local/sbin/k13-llm-switch
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
for d in "$OV_MODELS_DIR" "$STORAGE" "$SNAPSHOTS" "$K13_DATA_DIR" /opt/k13/qdrant "$BASE_DIR/logs"; do
  sudo mkdir -p "$d"
done
sudo chown -R "$USER_NAME:$USER_NAME" "$K13_DATA_DIR" "$OV_MODELS_DIR" "$STORAGE" "$SNAPSHOTS" /opt/k13 "$BASE_DIR/logs"

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
  "$BASE_DIR/.venv-openvino/bin/pip" install -q -r "$BASE_DIR/openvino/requirements.txt" \
    || warn "pip could not install every OpenVINO server package (no wheels for this Python?). Set OPENVINO_PYTHON in .env to the Python that already has OpenVINO, then rerun."
fi
missing="$("$OV_PY" - <<'PY' 2>/dev/null || true
import importlib.util
miss=[m for m in ("openvino","openvino_genai","fastapi","uvicorn","transformers","numpy","torch","huggingface_hub") if importlib.util.find_spec(m) is None]
print(" ".join(miss))
PY
)"
[[ -z "$missing" ]] && ok "OpenVINO Python has everything the server needs." \
  || warn "OpenVINO Python is missing: $missing  -> $OV_PY -m pip install -r $BASE_DIR/openvino/requirements.txt"

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

# ---- 6. systemd ----------------------------------------------------------------------------------
install_units
sudo systemctl enable k13-qdrant.service k13-openvino.service k13-web.service k13-rss-sync.timer >/dev/null 2>&1
sudo systemctl restart k13-qdrant.service k13-openvino.service
sudo systemctl restart k13-web.service
sudo systemctl start k13-rss-sync.timer

# ---- 7. Verify ---------------------------------------------------------------------------------------
log "Waiting for services (the first OpenVINO start downloads the models and can take 10+ minutes; this check only waits ~2 min)..."
for i in $(seq 1 30); do
  q=$(curl -fs --max-time 3 "http://127.0.0.1:${QDRANT_PORT}/healthz" >/dev/null 2>&1 && echo up || echo down)
  o=$(curl -fs --max-time 3 "http://127.0.0.1:${OV_PORT}/healthz" >/dev/null 2>&1 && echo up || echo down)
  [[ "$q" == up && "$o" == up ]] && break
  echo "    qdrant: $q  openvino: $o  ($i/30)"; sleep 4
done
if $WITH_SHARE; then "$BASE_DIR/scripts/setup-share.sh" || warn "Share setup failed; run ./scripts/setup-share.sh again after fixing the error above."; fi
if $WITH_BACKUPS; then "$BASE_DIR/scripts/install-cron.sh" || warn "Backup scheduling failed; run ./scripts/install-cron.sh again."; fi
echo
ok "Install finished. Next steps:"
echo "  - Status:        ./scripts/k13ctl.sh status"
echo "  - Watch models:  ./scripts/k13ctl.sh logs openvino   (first start downloads the LLM and embedding model)"
echo "  - Web portal:    http://$(hostname -I | awk '{print $1}'):$(env_get WEB_PORT 8501)"
$WITH_SHARE   || echo "  - Network share: ./scripts/setup-share.sh      (shares the backup folder over SMB)"
$WITH_BACKUPS || echo "  - Backups:       ./scripts/install-cron.sh     (nightly 04:00 / 04:15)"
echo "  (Log out and back in once so your shell picks up the render/video groups.)"
