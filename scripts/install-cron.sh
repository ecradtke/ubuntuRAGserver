#!/usr/bin/env bash
# ==============================================================================
# K13 AI Stack - install the nightly backup cron jobs (Debian)
#
# Usage:  ./scripts/install-cron.sh [--time HH:MM] [--keep N] [--remove]
#   --time HH:MM  Daily time for the project/Qdrant backup (default 04:00);
#                 the PostgreSQL backup runs 15 minutes later.
#   --keep N      Backups to retain per type (default 14)
#   --remove      Remove the K13 cron entries
#
# Installs entries in the CURRENT user's crontab (run as the user that owns /srv/dropit access).
# Re-running replaces the previous K13 entries; nothing else in the crontab is touched.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
TIME="04:00"; KEEP=14; REMOVE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --time)   TIME="${2:?--time needs HH:MM}"; shift 2 ;;
    --keep)   KEEP="${2:?--keep needs a number}"; shift 2 ;;
    --remove) REMOVE=true; shift ;;
    -h|--help) sed -n 2,13p "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[!] Unknown option: $1"; exit 1 ;;
  esac
done
[[ "$TIME" =~ ^([01][0-9]|2[0-3]):([0-5][0-9])$ ]] || { echo "[!] --time must be HH:MM (24h)"; exit 1; }
H1=$((10#${BASH_REMATCH[1]})); M1=$((10#${BASH_REMATCH[2]}))
T2=$(( (H1 * 60 + M1 + 15) % 1440 )); H2=$((T2 / 60)); M2=$((T2 % 60))

MARK_BEGIN="# >>> K13 backups (managed by scripts/install-cron.sh) >>>"
MARK_END="# <<< K13 backups <<<"
current="$(crontab -l 2>/dev/null || true)"
# drop any previous K13 block
cleaned="$(printf '%s\n' "$current" | sed "/^${MARK_BEGIN//\//\\/}\$/,/^${MARK_END//\//\\/}\$/d")"

if $REMOVE; then
  printf '%s\n' "$cleaned" | crontab -
  echo "[+] K13 cron entries removed."
  exit 0
fi

command -v flock >/dev/null || { echo "[!] flock not found (sudo apt install util-linux)"; exit 1; }
mkdir -p "${BASE_DIR}/logs"
block="${MARK_BEGIN}
${M1} ${H1} * * * flock -n /tmp/k13_backup.lock ${SCRIPT_DIR}/backup_k13.sh --keep ${KEEP} >> ${BASE_DIR}/logs/backup_k13.log 2>&1
${M2} ${H2} * * * flock -n /tmp/k13_pg_backup.lock ${SCRIPT_DIR}/backup_postgres.sh --keep ${KEEP} >> ${BASE_DIR}/logs/backup_postgres.log 2>&1
${MARK_END}"
{ printf '%s\n' "$cleaned" | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}'; printf '%s\n' "$block"; } | crontab -

printf '[+] Installed (daily at %02d:%02d project+Qdrant, %02d:%02d PostgreSQL, keep %s):\n' "$H1" "$M1" "$H2" "$M2" "$KEEP"
crontab -l | sed -n "/${MARK_BEGIN//\//\\/}/,/${MARK_END//\//\\/}/p"
echo "Logs: ${BASE_DIR}/logs/   |  Test now: ${SCRIPT_DIR}/backup_k13.sh"
