#!/usr/bin/env bash
# Remove the K13 services, units and the model-switch helper. Your data (models, Qdrant, .env) is kept.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
[[ $EUID -ne 0 ]] || die "Run as your normal user."
read -r -p "Stop and remove all K13 systemd units? Data stays. [y/N] " a; [[ "$a" == [yY]* ]] || exit 0
sudo systemctl disable --now k13-rss-sync.timer k13-web.service k13-openvino.service k13-qdrant.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/k13-*.service /etc/systemd/system/k13-*.timer
sudo systemctl daemon-reload
ok "Removed. Data is still in $(env_get K13_DATA_DIR /var/lib/k13) and /opt/k13 (delete by hand if you want it gone)."
