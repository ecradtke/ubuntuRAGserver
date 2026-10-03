#!/usr/bin/env bash
# Control the K13 services.
#   k13ctl.sh status
#   k13ctl.sh {start|stop|restart} {qdrant|openvino|web|rss|all}
#   k13ctl.sh logs {qdrant|openvino|web|rss}
#   k13ctl.sh test          embedding + chat smoke tests
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
load_k13_env

unit_of() {
  case "$1" in
    qdrant) echo k13-qdrant.service ;; openvino) echo k13-openvino.service ;;
    web) echo k13-web.service ;; rss) echo k13-rss-sync.timer ;;
    *) die "unknown service '$1' (qdrant|openvino|web|rss|all)" ;;
  esac
}
ALL=(qdrant openvino web rss)

case "${1:-status}" in
  status)
    for s in "${ALL[@]}"; do
      u="$(unit_of "$s")"; printf '%-8s %-10s' "$s" "$(systemctl is-active "$u" 2>/dev/null)"
      printf ' (%s)\n' "$(systemctl is-enabled "$u" 2>/dev/null)"
    done
    echo
    echo "--- health ---"
    printf 'qdrant : '; curl -fs --max-time 3 "http://127.0.0.1:${QDRANT_PORT}/healthz" || echo "not responding"; echo
    printf 'openvino: '; curl -fs --max-time 20 "http://127.0.0.1:${OV_PORT}/healthz" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["status"], "| LLM:", d.get("llm_hardware"), d.get("current_model"), "| embeddings:", d.get("embed_hardware"))' 2>/dev/null || echo "not responding (still loading models?)"
    ;;
  start|stop|restart)
    act="$1"; target="${2:-}"; [[ -n "$target" ]] || die "usage: $0 $act {qdrant|openvino|web|rss|all}"
    if [[ "$target" == all ]]; then for s in "${ALL[@]}"; do sudo systemctl "$act" "$(unit_of "$s")"; done
    else sudo systemctl "$act" "$(unit_of "$target")"; fi
    ;;
  logs)
    [[ -n "${2:-}" ]] || die "usage: $0 logs {qdrant|openvino|web|rss}"
    u="$(unit_of "$2")"; [[ "$2" == rss ]] && u=k13-rss-sync.service
    journalctl -fu "$u" -n 100
    ;;
  test)
    log "Embedding test..."
    curl -fs "http://127.0.0.1:${OV_PORT}/api/embeddings" -d '{"prompt":"High-NA EUV lithography"}' \
      | python3 -c 'import sys,json; v=json.load(sys.stdin)["embedding"]; print("  vector dimensions:", len(v))' || warn "embedding test failed"
    log "Chat test..."
    curl -fs "http://127.0.0.1:${OV_PORT}/api/generate" -d '{"prompt":"Reply with the single word: ready","stream":false}' \
      | python3 -c 'import sys,json; print("  ", json.load(sys.stdin)["response"].strip()[:200])' || warn "chat test failed"
    ;;
  *) sed -n 2,8p "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
