#!/usr/bin/env bash
# ==============================================================================
# K13 AI Stack — Backup Script
# Backs up the Qdrant vector database + the project folder into a single
# compressed archive on the shared folder "dropit".
#
# Usage:  ./scripts/backup_k13.sh [--no-env] [--keep N] [--dest DIR]
#   --no-env    Leave .env (passwords/tokens) out of the archive
#   --keep N    Keep only the newest N backups in dropit (default: keep all)
#   --dest DIR  Override the dropit path (default: $DROPIT_DIR or /srv/dropit)
#
# Output: <dropit>/fetch42_backup_YYYYmmdd_HHMMSS.tar.gz
#   fetch42/          project folder (minus .git, venvs, caches, model weights)
#   qdrant_snapshot/  Qdrant full-storage snapshot (or raw storage dir fallback)
#
# Restore: see the "restore.txt" bundled in the archive.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(dirname "$SCRIPT_DIR")}"
ENV_FILE="${BASE_DIR}/.env"

# Read only the keys we need from .env. The file is parsed, never sourced, so
# unquoted values with spaces (e.g. app passwords) can't be executed as commands.
env_get() {
  local v
  [[ -f "$ENV_FILE" ]] || return 0
  v="$(grep -E "^[[:space:]]*$1=" "$ENV_FILE" | tail -n1 | cut -d= -f2- || true)"
  v="${v%%[[:space:]]#*}"                      # drop trailing "  # comment"
  v="${v%"${v##*[![:space:]]}"}"               # trim trailing whitespace
  v="${v#"${v%%[![:space:]]*}"}"               # trim leading whitespace
  v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"   # strip quotes
  printf '%s' "$v"
}
for k in DROPIT_DIR QDRANT_PORT QDRANT_STORAGE_DIR QDRANT_SNAPSHOTS_DIR NAS_HOST; do
  [[ -z "${!k:-}" ]] && export "$k=$(env_get "$k")"
done

DROPIT_DIR="${DROPIT_DIR:-/srv/dropit}"
QDRANT_URL="${QDRANT_URL:-http://localhost:${QDRANT_PORT:-6333}}"
QDRANT_STORAGE="${QDRANT_STORAGE_DIR:-/var/lib/k13/qdrant/storage}"
QDRANT_SNAPSHOTS="${QDRANT_SNAPSHOTS_DIR:-/var/lib/k13/qdrant/snapshots}"
INCLUDE_ENV=true
KEEP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-env) INCLUDE_ENV=false; shift ;;
    --keep)   KEEP="${2:?--keep needs a number}"; shift 2 ;;
    --dest)   DROPIT_DIR="${2:?--dest needs a path}"; shift 2 ;;
    -h|--help) sed -n 2,17p "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[!] Unknown option: $1"; exit 1 ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_NAME="fetch42_backup_${STAMP}.tar.gz"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "=================================================="
echo "  K13 AI Stack: Backup"
echo "  Project:  ${BASE_DIR}"
echo "  Target:   ${DROPIT_DIR}/${ARCHIVE_NAME}"
echo "=================================================="

# 1. Make sure the shared folder is reachable
if [[ ! -d "$DROPIT_DIR" ]]; then
  echo "[!] ERROR: dropit folder not found at: ${DROPIT_DIR}"
  echo "    Mount the share, or set DROPIT_DIR in .env / pass --dest."
  echo "    e.g. sudo mount -t cifs //${NAS_HOST:-192.168.4.22}/dropit ${DROPIT_DIR} -o credentials=/root/.smbcred"
  exit 1
fi
if [[ ! -w "$DROPIT_DIR" ]]; then
  echo "[!] ERROR: ${DROPIT_DIR} is not writable by $(id -un)."
  exit 1
fi

# 2. Qdrant: take a consistent full-storage snapshot via the REST API
mkdir -p "${STAGE}/qdrant_snapshot"
echo "[*] Requesting Qdrant snapshot from ${QDRANT_URL}..."
SNAP_NAME=""
if curl -fsS --max-time 5 "${QDRANT_URL}/healthz" >/dev/null 2>&1; then
  SNAP_NAME="$(curl -fsS -X POST "${QDRANT_URL}/snapshots?wait=true" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["name"])' || true)"
fi

if [[ -n "$SNAP_NAME" && -f "${QDRANT_SNAPSHOTS}/${SNAP_NAME}" ]]; then
  cp "${QDRANT_SNAPSHOTS}/${SNAP_NAME}" "${STAGE}/qdrant_snapshot/"
  # Remove the server-side copy so snapshots don't pile up on disk
  curl -fsS -X DELETE "${QDRANT_URL}/snapshots/${SNAP_NAME}?wait=true" >/dev/null 2>&1 || true
  echo "[+] Snapshot ${SNAP_NAME} captured."
elif [[ -n "$SNAP_NAME" ]]; then
  # Snapshot exists but isn't on the host path; pull it over HTTP instead
  curl -fsS -o "${STAGE}/qdrant_snapshot/${SNAP_NAME}" "${QDRANT_URL}/snapshots/${SNAP_NAME}"
  curl -fsS -X DELETE "${QDRANT_URL}/snapshots/${SNAP_NAME}?wait=true" >/dev/null 2>&1 || true
  echo "[+] Snapshot ${SNAP_NAME} downloaded over HTTP."
elif [[ -d "$QDRANT_STORAGE" ]]; then
  echo "[!] Qdrant API unavailable — falling back to raw copy of ${QDRANT_STORAGE}."
  echo "    (Stop k13-qdrant first if you need this copy to be crash-consistent.)"
  cp -a "$QDRANT_STORAGE" "${STAGE}/qdrant_snapshot/storage"
else
  echo "[!] WARNING: no Qdrant data found (API down, ${QDRANT_STORAGE} missing)."
  echo "    Continuing with project files only."
fi

# 3. Restore notes bundled with the archive
cat > "${STAGE}/restore.txt" <<EOF
Backup created ${STAMP} on $(hostname)
Project files : extract, then copy fetch42/ to /opt/fetch42 (or anywhere).
Qdrant (snapshot .snapshot file):
  curl -X POST "http://localhost:6333/snapshots/upload?priority=snapshot" \\
       -H "Content-Type:multipart/form-data" -F "snapshot=@qdrant_snapshot/<file>.snapshot"
  (or start qdrant with: qdrant --storage-snapshot <file>.snapshot)
Qdrant (raw storage/ fallback): stop k13-qdrant, replace \$QDRANT_STORAGE_DIR
  with qdrant_snapshot/storage, start it again.
EOF

# 4. Build the archive locally first, then move it onto the share
EXCLUDES=(
  --exclude=./.git
  --exclude=.venv --exclude=.venv-openvino --exclude=venv --exclude=__pycache__ --exclude='*.pyc'
  --exclude=openvino_models --exclude=logs
)
$INCLUDE_ENV || EXCLUDES+=(--exclude=./.env)

# Live Qdrant data is captured by the snapshot above. If it sits inside the
# project folder, keep the raw (container-owned, in-use) files out of the tar.
for d in "$QDRANT_STORAGE" "$QDRANT_SNAPSHOTS"; do
  [[ "$d" == "$BASE_DIR"/* ]] && EXCLUDES+=(--exclude="./${d#"$BASE_DIR"/}")
done

echo "[*] Compressing project + vector DB..."
LOCAL_ARCHIVE="${STAGE}/${ARCHIVE_NAME}"
tar -czf "$LOCAL_ARCHIVE" "${EXCLUDES[@]}" \
  --transform='s,^\.,fetch42,' -C "$BASE_DIR" . \
  -C "$STAGE" qdrant_snapshot restore.txt

# 5. Copy to dropit (temp name first so readers never see a partial file)
cp "$LOCAL_ARCHIVE" "${DROPIT_DIR}/.${ARCHIVE_NAME}.part"
mv "${DROPIT_DIR}/.${ARCHIVE_NAME}.part" "${DROPIT_DIR}/${ARCHIVE_NAME}"
chmod 600 "${DROPIT_DIR}/${ARCHIVE_NAME}" 2>/dev/null || true   # may hold .env secrets
tar -tzf "${DROPIT_DIR}/${ARCHIVE_NAME}" >/dev/null   # verify readable

echo "[+] Backup written: ${DROPIT_DIR}/${ARCHIVE_NAME} ($(du -h "${DROPIT_DIR}/${ARCHIVE_NAME}" | cut -f1))"
$INCLUDE_ENV && echo "[!] Archive contains .env credentials — keep the share access-restricted (or use --no-env)."

# 6. Optional retention
if [[ "$KEEP" -gt 0 ]]; then
  mapfile -t OLD < <(ls -1t "${DROPIT_DIR}"/fetch42_backup_*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)))
  for f in "${OLD[@]}"; do
    echo "[*] Pruning old backup: $(basename "$f")"
    rm -f "$f"
  done
fi
echo "[+] Done."
