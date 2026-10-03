#!/usr/bin/env bash
# systemd ExecStart for k13-qdrant: native Qdrant binary with settings from .env
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
load_k13_env
QDRANT_DIR="${QDRANT_DIR:-/opt/k13/qdrant}"
[[ -x "$QDRANT_DIR/qdrant" ]] || die "Qdrant is not installed at $QDRANT_DIR (run scripts/install.sh)"
STORAGE="$(env_get QDRANT_STORAGE_DIR "$K13_DATA_DIR/qdrant/storage")"
SNAPSHOTS="$(env_get QDRANT_SNAPSHOTS_DIR "$K13_DATA_DIR/qdrant/snapshots")"
mkdir -p "$STORAGE" "$SNAPSHOTS"
export QDRANT__STORAGE__STORAGE_PATH="$STORAGE"
export QDRANT__STORAGE__SNAPSHOTS_PATH="$SNAPSHOTS"
export QDRANT__SERVICE__HOST="$(env_get QDRANT_BIND 0.0.0.0)"
export QDRANT__SERVICE__HTTP_PORT="$QDRANT_PORT"
export QDRANT__SERVICE__ENABLE_STATIC_CONTENT=true
export QDRANT__SERVICE__STATIC_CONTENT_DIR="$QDRANT_DIR/static"
export QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=64
export QDRANT__TELEMETRY_DISABLED=true
cd "$QDRANT_DIR"
exec "$QDRANT_DIR/qdrant"
