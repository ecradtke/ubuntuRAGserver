#!/usr/bin/env bash
# ==============================================================================
# K13 AI Stack - create the "dropit" backup folder and share it on the network (Samba/SMB)
#
# Usage: ./scripts/setup-share.sh [--path DIR] [--user NAME] [--subnet CIDR] [--remove]
#   --path DIR      folder to share (default: DROPIT_DIR from .env, else /srv/dropit)
#   --user NAME     the one user allowed to log in (default: you)
#   --subnet CIDR   networks allowed to connect (default: auto-detected LAN, e.g. 192.168.4.0/22)
#   --remove        remove the share definition (the folder and its files are kept)
# Safe to run again. You will be asked once to choose the Samba password (separate from your login).
# ==============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

SUDO="${SUDO-sudo}"                                  # tests set SUDO= to run without root
SMB_CONF="${K13_SMB_CONF:-/etc/samba/smb.conf}"
SHARE_PATH="$(env_get DROPIT_DIR /srv/dropit)"; SHARE_USER="$(id -un)"; SUBNET=""; REMOVE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --path)   SHARE_PATH="${2:?--path needs a folder}"; shift 2 ;;
    --user)   SHARE_USER="${2:?--user needs a name}"; shift 2 ;;
    --subnet) SUBNET="${2:?--subnet needs a CIDR}"; shift 2 ;;
    --remove) REMOVE=true; shift ;;
    -h|--help) sed -n 2,11p "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done
[[ "$SHARE_PATH" == /* ]] || die "--path must be an absolute path"
[[ "$SHARE_PATH" =~ ^[A-Za-z0-9._/+-]+$ ]] || die "the folder name may only contain letters, digits and . _ / + -"

MARK_BEGIN="# >>> K13 dropit share (managed by scripts/setup-share.sh) >>>"
MARK_END="# <<< K13 dropit share <<<"

# delete our previous block from the config text on stdin
strip_block() { awk -v b="$MARK_BEGIN" -v e="$MARK_END" '$0==b{skip=1} !skip{print} $0==e{skip=0}'; }

if $REMOVE; then
  [[ -f "$SMB_CONF" ]] || die "$SMB_CONF not found"
  strip_block < "$SMB_CONF" | $SUDO tee "$SMB_CONF.new" >/dev/null && $SUDO mv "$SMB_CONF.new" "$SMB_CONF"
  [[ -n "${K13_SHARE_TEST:-}" ]] || $SUDO systemctl restart smbd
  ok "Share removed from $SMB_CONF. The folder $SHARE_PATH and its files were not touched."
  exit 0
fi

# ---- work out the LAN to allow ---------------------------------------------------------------
if [[ -z "$SUBNET" ]]; then
  SUBNET="$(python3 - <<'PY' 2>/dev/null || true
import ipaddress, re, subprocess
try:
    dev = re.search(r"dev (\S+)", subprocess.check_output(["ip", "-4", "route", "get", "1.1.1.1"], text=True)).group(1)
    line = subprocess.check_output(["ip", "-4", "-o", "addr", "show", "dev", dev], text=True)
    cidr = re.search(r"inet (\S+)", line).group(1)
    print(ipaddress.ip_interface(cidr).network)
except Exception:
    pass
PY
)"
fi
[[ -n "$SUBNET" ]] || die "Could not detect your network. Pass it yourself, for example: --subnet 192.168.4.0/24"
[[ "$SUBNET" =~ ^[0-9.]+/[0-9]+$ ]] || die "--subnet must look like 192.168.4.0/24"
log "Sharing $SHARE_PATH with user '$SHARE_USER' on network $SUBNET"

# ---- packages and folder ------------------------------------------------------------------------
if [[ -z "${K13_SHARE_TEST:-}" ]]; then
  if findmnt -rn "$SHARE_PATH" >/dev/null 2>&1; then
    die "$SHARE_PATH is already a mount point (a mounted NAS share?). Unmount it and remove its /etc/fstab line first."
  fi
  log "Installing Samba..."
  $SUDO apt-get install -y -qq samba smbclient >/dev/null
  id "$SHARE_USER" >/dev/null 2>&1 || die "user '$SHARE_USER' does not exist"
  $SUDO mkdir -p "$SHARE_PATH"
  $SUDO chown "$SHARE_USER:$SHARE_USER" "$SHARE_PATH"
  $SUDO chmod 770 "$SHARE_PATH"
fi

# ---- Samba config: back it up once, then (re)write our block ---------------------------------------------
[[ -f "$SMB_CONF" ]] || die "$SMB_CONF not found (is Samba installed?)"
[[ -f "$SMB_CONF.k13-backup" ]] || $SUDO cp "$SMB_CONF" "$SMB_CONF.k13-backup"
tmp="$(mktemp)"
strip_block < "$SMB_CONF" > "$tmp"
# hardening in [global]: SMB2+ only, no guest logins (added only if the file does not already set them)
grep -qiE '^[[:space:]]*server min protocol' "$tmp" || sed -i '0,/^\[global\]/s//[global]\n   server min protocol = SMB2/' "$tmp"
grep -qiE '^[[:space:]]*map to guest' "$tmp" || sed -i '0,/^\[global\]/s//[global]\n   map to guest = never/' "$tmp"
cat >> "$tmp" <<BLOCK
$MARK_BEGIN
[dropit]
   path = $SHARE_PATH
   browseable = yes
   read only = no
   valid users = $SHARE_USER
   create mask = 0660
   directory mask = 0770
   hosts allow = $SUBNET 127.0.0.1
   hosts deny = ALL
$MARK_END
BLOCK
$SUDO install -m 644 -o root -g root "$tmp" "$SMB_CONF"; rm -f "$tmp"
if [[ -z "${K13_SHARE_TEST:-}" ]]; then
  testparm -s >/dev/null 2>&1 || { $SUDO cp "$SMB_CONF.k13-backup" "$SMB_CONF"; die "Samba rejected the new config; the original was restored."; }
fi

# ---- Samba password, service, firewall ------------------------------------------------------------------------
if [[ -z "${K13_SHARE_TEST:-}" ]]; then
  if $SUDO pdbedit -L 2>/dev/null | grep -q "^${SHARE_USER}:"; then
    ok "Samba user '$SHARE_USER' already exists (change the password any time with: sudo smbpasswd $SHARE_USER)"
  else
    echo "Choose the password for the network share (it is separate from your Linux login):"
    $SUDO smbpasswd -a "$SHARE_USER"
  fi
  $SUDO systemctl enable --now smbd >/dev/null 2>&1
  $SUDO systemctl restart smbd
  if command -v ufw >/dev/null && $SUDO ufw status 2>/dev/null | grep -q "Status: active"; then
    $SUDO ufw allow from "$SUBNET" to any app Samba >/dev/null && ok "Firewall: allowed Samba from $SUBNET"
  fi
fi
ok "Share ready: \\\\$(hostname -I 2>/dev/null | awk '{print $1}')\\dropit  (Windows)   smb://$(hostname -I 2>/dev/null | awk '{print $1}')/dropit  (Mac)"
echo "    Sign in as '$SHARE_USER' with the Samba password. Backups land in $SHARE_PATH (DROPIT_DIR in .env)."
