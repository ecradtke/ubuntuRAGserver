# K13 AI Hub - Ubuntu Server Edition (native, no containers)

A locally hosted RAG platform on Intel hardware, running as plain **systemd services** on Ubuntu Server:

```
 browser ──► k13-web (Streamlit :8501) ──┬──► k13-llama  (llama.cpp llama-server :8080)  chat LLM, GGUF models, Intel GPU
                                         ├──► k13-embed  (OpenVINO server :11434)         embeddings on the Intel NPU
                                         └──► k13-qdrant (Qdrant :6333)                   vector database

 k13-rss-sync.timer (every 5 min): FreshRSS (NAS) ──► PostgreSQL (NAS) ──► embeddings ──► Qdrant
```

| Piece | What it is | Port |
|---|---|---|
| `k13-llama` | your llama.cpp `llama-server`, started with the model you pick | 8080 (this server only) |
| `k13-embed` | OpenVINO embedding server (`embed/embed_server.py`), NPU with CPU fallback, 384-dim `bge-small` | 11434 (this server only) |
| `k13-qdrant` | native Qdrant binary installed by the installer | 6333 / 6334 |
| `k13-web` | Streamlit portal (chat, search, model switcher, precise time-range filter) | 8501 |
| `k13-rss-sync` | one-shot sync run by a timer every 5 minutes | - |

Why llama.cpp for chat: GGUF models can be split between GPU and CPU (`LLAMA_GPU_LAYERS`), which lifts the
~6.8 GB GPU-memory ceiling that OpenVINO's LLM pipeline hit on Lunar Lake.

## Quick start

```bash
sudo mkdir -p /opt/fetch42 && sudo chown $USER:$USER /opt/fetch42
git clone https://github.com/ecradtke/ubuntuport /opt/fetch42 && cd /opt/fetch42
./scripts/install.sh          # first run creates .env and stops: edit it, then run again
nano .env
./scripts/install.sh
# get a chat model: replace REPO and FILE with real names from huggingface.co (see docs/install_ubuntu.md)
./scripts/download-model.sh REPO FILE.gguf --switch
./scripts/k13ctl.sh status
./scripts/install-cron.sh     # nightly backups at 04:00 / 04:15
```

Full guide (prerequisites, GPU/NPU notes, models, troubleshooting): [docs/install_ubuntu.md](docs/install_ubuntu.md).

## Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `install.sh` | Idempotent installer: packages, folders, Python envs, Qdrant, systemd units |
| `update.sh` | `git pull`, refresh packages and units, restart services |
| `k13ctl.sh` | `status`, `start/stop/restart <svc>`, `logs <svc>`, `models`, `test` |
| `download-model.sh` | Download a GGUF from Hugging Face (resumable); `--switch` activates it |
| `llm-switch.sh` | Installed as `/usr/local/sbin/k13-llm-switch`; the portal uses it to change models |
| `backup_k13.sh`, `backup_postgres.sh`, `install-cron.sh` | Project + Qdrant and PostgreSQL backups to `DROPIT_DIR`, scheduled by cron |
| `run-*.sh`, `_lib.sh` | What systemd actually launches; read settings from `.env` |

## Notes
- Settings live only in `.env` (gitignored, chmod 600). Quote values with spaces. Re-run `install.sh` after editing paths.
- Developer rules for AI assistants: [AGENTS.md](AGENTS.md).
