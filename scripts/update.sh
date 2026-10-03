#!/usr/bin/env bash
# Pull the latest code, refresh Python packages and units, and restart the services.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
cd "$BASE_DIR"
log "Pulling latest code..."
git pull --ff-only
log "Updating Python packages..."
"$BASE_DIR/.venv/bin/pip" install -q -U -r web/requirements.txt
OV_PY="$(env_get OPENVINO_PYTHON)"; OV_PY="${OV_PY:-$BASE_DIR/.venv-openvino/bin/python}"
"$OV_PY" -m pip install -q -r openvino/requirements.txt || warn "OpenVINO server packages not updated"
"$BASE_DIR/scripts/install.sh" --units-only
sudo systemctl restart k13-openvino.service k13-web.service
sudo systemctl try-restart k13-qdrant.service
ok "Updated. Check with: ./scripts/k13ctl.sh status"
