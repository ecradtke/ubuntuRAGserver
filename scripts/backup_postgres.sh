#!/usr/bin/env bash
# ==============================================================================
# K13 AI Stack — PostgreSQL Backup Script
# Dumps the PostgreSQL database(s) on the NAS (PG_HOST:PG_PORT from .env) and
# writes a compressed archive to the shared folder "dropit".
#
# Usage:  ./scripts/backup_postgres.sh [--db NAME]... [--keep N] [--dest DIR]
#   --db NAME   Database to dump (repeatable). Default: fetch42 and freshrss
#   --keep N    Keep only the newest N postgres backups in dropit (default: all)
#   --dest DIR  Override the dropit path (default: $DROPIT_DIR or /srv/dropit)
#
# Output: <dropit>/fetch42_postgres_YYYYmmdd_HHMMSS.tar.gz containing one
#         <db>.dump per database (pg_dump custom format, already compressed)
#         plus restore.txt.
#
# Requires the PostgreSQL client tools (sudo apt install postgresql-client).
# pg_dump must be the same major version as the server or newer.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(dirname "$SCRIPT_DIR")}"
ENV_FILE="${BASE_DIR}/.env"

# Read only the keys we need from .env (parsed, never sourced).
env_get() {
  local v
  [[ -f "$ENV_FILE" ]] || return 0
  v="$(grep -E "^[[:space:]]*$1=" "$ENV_FILE" | tail -n1 | cut -d= -f2- || true)"
  v="${v%%[[:space:]]#*}"
  v="${v%"${v##*[![:space:]]}"}"
  v="${v#"${v%%[![:space:]]*}"}"
  v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"
  printf '%s' "$v"
}
for k in DROPIT_DIR PG_HOST PG_PORT PG_DB PG_USER PG_PASS NAS_HOST; do
  [[ -z "${!k:-}" ]] && export "$k=$(env_get "$k")"
done

DROPIT_DIR="${DROPIT_DIR:-/srv/dropit}"
PG_HOST="${PG_HOST:-${NAS_HOST:-192.168.4.22}}"
PG_PORT="${PG_PORT:-5433}"
PG_USER="${PG_USER:-freshrss_user}"
DBS=()
KEEP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db)   DBS+=("${2:?--db needs a name}"); shift 2 ;;
    --keep) KEEP="${2:?--keep needs a number}"; shift 2 ;;
    --dest) DROPIT_DIR="${2:?--dest needs a path}"; shift 2 ;;
    -h|--help) sed -n 2,18p "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[!] Unknown option: $1"; exit 1 ;;
  esac
done

# Password goes through a private temp .pgpass, never on the command line
if [[ -z "${PG_PASS:-}" ]]; then
  echo "[!] ERROR: PG_PASS is not set in ${ENV_FILE}."
  exit 1
fi
if ! command -v pg_dump >/dev/null; then
  echo "[!] ERROR: pg_dump not found. Install it: sudo apt install postgresql-client"
  exit 1
fi
if [[ ! -d "$DROPIT_DIR" || ! -w "$DROPIT_DIR" ]]; then
  echo "[!] ERROR: dropit folder ${DROPIT_DIR} is missing or not writable."
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_NAME="fetch42_postgres_${STAMP}.tar.gz"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

PGPASSFILE="${STAGE}/.pgpass"
# .pgpass format is host:port:db:user:password, with ':' and '\' escaped
esc() { printf '%s' "$1" | sed 's/[\\:]/\\&/g'; }
printf '%s:%s:*:%s:%s\n' "$(esc "$PG_HOST")" "$PG_PORT" "$(esc "$PG_USER")" "$(esc "$PG_PASS")" > "$PGPASSFILE"
chmod 600 "$PGPASSFILE"
export PGPASSFILE
export PGCONNECT_TIMEOUT=10
PGARGS=(-h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER")

echo "=================================================="
echo "  K13 AI Stack: PostgreSQL Backup"
echo "  Server:  ${PG_USER}@${PG_HOST}:${PG_PORT}"
echo "  Target:  ${DROPIT_DIR}/${ARCHIVE_NAME}"
echo "=================================================="

# Only these databases are backed up unless --db is given
[[ ${#DBS[@]} -eq 0 ]] && DBS=(fetch42 freshrss)

mkdir -p "${STAGE}/pg"
for db in "${DBS[@]}"; do
  echo "[*] Dumping database: ${db}"
  pg_dump "${PGARGS[@]}" -d "$db" -Fc -Z 6 --no-password -f "${STAGE}/pg/${db}.dump"
  echo "[+] ${db}.dump ($(du -h "${STAGE}/pg/${db}.dump" | cut -f1))"
done

cat > "${STAGE}/pg/restore.txt" <<EOF
Backup created ${STAMP} from ${PG_HOST}:${PG_PORT} (databases: ${DBS[*]})
Restore one database (create it first if it doesn't exist):
  createdb -h HOST -p PORT -U USER <db>
  pg_restore -h HOST -p PORT -U USER -d <db> --no-owner --clean --if-exists <db>.dump
List contents without restoring:  pg_restore -l <db>.dump
EOF

tar -czf "${STAGE}/${ARCHIVE_NAME}" -C "${STAGE}/pg" .
cp "${STAGE}/${ARCHIVE_NAME}" "${DROPIT_DIR}/.${ARCHIVE_NAME}.part"
mv "${DROPIT_DIR}/.${ARCHIVE_NAME}.part" "${DROPIT_DIR}/${ARCHIVE_NAME}"
chmod 600 "${DROPIT_DIR}/${ARCHIVE_NAME}" 2>/dev/null || true
tar -tzf "${DROPIT_DIR}/${ARCHIVE_NAME}" >/dev/null

echo "[+] Backup written: ${DROPIT_DIR}/${ARCHIVE_NAME} ($(du -h "${DROPIT_DIR}/${ARCHIVE_NAME}" | cut -f1))"

if [[ "$KEEP" -gt 0 ]]; then
  mapfile -t OLD < <(ls -1t "${DROPIT_DIR}"/fetch42_postgres_*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)))
  for f in "${OLD[@]}"; do
    echo "[*] Pruning old backup: $(basename "$f")"
    rm -f "$f"
  done
fi
echo "[+] Done."
