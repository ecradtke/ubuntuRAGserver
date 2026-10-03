# System Architecture & Platform Guidelines: K13 AI Hub on Ubuntu Server (native)

## Platform directive
This deployment runs **natively on Ubuntu Server** (minimal desktop) with **systemd**. There are **no containers**:
never suggest Docker, Compose or `host.docker.internal`. It is **not Ollama**. Use bash, systemd and Linux paths.

## Services (all systemd units, run as the installing user)
| Unit | Role | Default address |
|---|---|---|
| `k13-llama.service` | llama.cpp `llama-server`: chat LLM, GGUF models, OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`, `/props`) | `127.0.0.1:8080` |
| `k13-embed.service` | `embed/embed_server.py`: OpenVINO embeddings on the Intel NPU (CPU fallback); Ollama-style `/api/embeddings`, OpenAI `/v1/embeddings`, `/healthz` | `127.0.0.1:11434` |
| `k13-qdrant.service` | Native Qdrant binary in `/opt/k13/qdrant`, data in `K13_DATA_DIR/qdrant` | `:6333` |
| `k13-web.service` | Streamlit portal `web/app.py` | `:8501` |
| `k13-rss-sync.timer` | Runs `rss/sync_freshrss.py` every 5 minutes (FreshRSS -> Postgres -> Qdrant) | - |

External: FreshRSS API and PostgreSQL (`fetch42`, `freshrss`) on the NAS; backups go to `DROPIT_DIR`.

## Layout
`embed/` OpenVINO embedding server - `web/` portal - `rss/` sync - `systemd/` unit templates (`@USER@`, `@REPO@`) -
`scripts/` installer, control, model, backup scripts - `docs/` guides. Config is a single `.env` parsed (never sourced) by `scripts/_lib.sh`;
the `scripts/run-*.sh` wrappers that systemd launches read it, so a service restart picks up edits.

## Rules for agents
1. **Chat LLM = llama.cpp, embeddings = OpenVINO.** Do not reintroduce OpenVINO LLM pipelines or the Ollama API for chat.
2. **Embedding dimension must match the Qdrant collection** (384 for `bge-small-en-v1.5`, 768 for `bge-base-en-v1.5`).
3. **Model switching**: the portal calls `sudo -n /usr/local/sbin/k13-llm-switch FILE.gguf` (root-owned, baked paths,
   strictly validated name, sudoers rule for that single script). Keep it that way; never give the web user broader sudo.
4. **Never put secrets in git.** Credentials live only in `.env`. `.env`: comments on their own line, no inline `# ...`; quote values with spaces.
5. systemd units are templates in `systemd/`; `scripts/install.sh --units-only` renders them. Services start via `scripts/run-*.sh`.
6. Qdrant URL/hosts: the web portal reads `QDRANT_URL`, the sync reads `QDRANT_HOST`/`QDRANT_PORT` (derived in `run-rss-sync.sh`).
