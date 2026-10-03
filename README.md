# K13 AI Hub - Ubuntu Server Edition (native OpenVINO, no containers)

A locally hosted RAG platform on Intel hardware, running as plain **systemd services** on Ubuntu Server.
**OpenVINO does all the AI work**: the chat LLM runs on the Intel GPU and the embeddings on the Intel NPU.

```
 browser ──► k13-web (Streamlit :8501) ──┬──► k13-openvino (OpenVINO server :11434)   LLM on GPU + embeddings on NPU
                                         └──► k13-qdrant   (Qdrant :6333)             vector database

 k13-rss-sync.timer (every 5 min): FreshRSS (NAS) ──► PostgreSQL (NAS) ──► embeddings ──► Qdrant
```

| Piece | What it is | Port |
|---|---|---|
| `k13-openvino` | `openvino/server.py`: Ollama- and OpenAI-compatible API, LLM via `openvino_genai` (GPU), `bge` embeddings (NPU, CPU fallback) | 11434 (this server only by default) |
| `k13-qdrant` | native Qdrant binary installed by the installer | 6333 / 6334 |
| `k13-web` | Streamlit portal (chat, search, LLM and embedding model switchers, precise time-range filter) | 8501 |
| `k13-rss-sync` | one-shot sync run by a timer every 5 minutes | - |

## Quick start

```bash
sudo mkdir -p /opt/fetch42 && sudo chown $USER:$USER /opt/fetch42
git clone https://github.com/ecradtke/ubuntuport /opt/fetch42 && cd /opt/fetch42
./scripts/install.sh          # first run creates .env and stops: edit it, then run again
nano .env
./scripts/install.sh
./scripts/k13ctl.sh logs openvino     # first start downloads and compiles the models
./scripts/k13ctl.sh status
./scripts/install-cron.sh             # nightly backups at 04:00 / 04:15
```

Full guide (prerequisites, GPU/NPU notes, models, troubleshooting): [docs/install_ubuntu.md](docs/install_ubuntu.md).

## Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `install.sh` | Idempotent installer: packages, folders, Python envs, Qdrant, systemd units |
| `update.sh` | `git pull`, refresh packages and units, restart services |
| `k13ctl.sh` | `status`, `start/stop/restart <svc>`, `logs <svc>`, `test` |
| `backup_k13.sh`, `backup_postgres.sh`, `install-cron.sh` | Project + Qdrant and PostgreSQL backups to `DROPIT_DIR`, scheduled by cron |
| `uninstall.sh` | Remove the services (data kept) |
| `run-*.sh`, `_lib.sh` | What systemd actually launches; read settings from `.env` |

## Notes
- Settings live only in `.env` (gitignored, chmod 600). Quote values with spaces. Re-run `install.sh --units-only` after changing the project path or user.
- The GPU runtime on Lunar Lake reports about 6.8 GB usable, so **Qwen3-8B int4 is the largest LLM that fits** on the GPU. A switch to a model that only fits on the CPU is rolled back (set `SWITCH_ALLOW_CPU=on` to allow it).
- Developer rules for AI assistants: [AGENTS.md](AGENTS.md).
