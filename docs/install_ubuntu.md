# K13 AI Stack - Install Guide (Ubuntu Server, native)

You already have **OpenVINO** and **llama.cpp** installed on this machine. This guide wires everything else around them.

## 0. Check what you have
```bash
command -v llama-server && llama-server --version          # llama.cpp server binary
ls -l /dev/accel /dev/dri                                  # NPU (accel0) and GPU (renderD128)
python3 -c "import openvino as ov; print(ov.Core().available_devices)"   # expect ['CPU','GPU','NPU']
```
- If OpenVINO lives in a venv, note that Python's path (e.g. `/home/e/ov/bin/python`). You will put it in `.env` as `OPENVINO_PYTHON`.
- If OpenVINO came from an archive that needs `source .../setupvars.sh`, note that script's path (`OPENVINO_SETUPVARS`).
- If llama.cpp was built with the **SYCL** backend you normally run `source /opt/intel/oneapi/setvars.sh` first: put that line in a small script
  and set `LLAMA_PRE_SCRIPT` to it. **Vulkan** builds need nothing extra.
- Your user must be in the `render` and `video` groups (the installer adds them; the services get them automatically).

## 1. Get the project
```bash
sudo mkdir -p /opt/fetch42 && sudo chown $USER:$USER /opt/fetch42
git clone https://github.com/ecradtke/ubuntuport /opt/fetch42
cd /opt/fetch42
```
A private repo asks for a username and a **personal access token** (fine-grained, `Contents: Read-only`) as the password.

## 2. Configure and install
```bash
./scripts/install.sh        # first run creates .env and stops
nano .env                   # see the table below
./scripts/install.sh        # second run installs everything
```
Set at least:

| Setting | Meaning |
|---|---|
| `LLAMA_SERVER_BIN` | path from `command -v llama-server` |
| `OPENVINO_PYTHON` / `OPENVINO_SETUPVARS` | only if OpenVINO is not in this project's own venv (blank = the installer creates `.venv-openvino`) |
| `FRESHRSS_*`, `PG_*` | your NAS FreshRSS and PostgreSQL logins |
| `HF_TOKEN` | Hugging Face token (embedding model export, gated GGUF repos) |
| `DROPIT_DIR` | backup folder (default `/srv/dropit`) |

What the installer does: installs OS packages, creates `/var/lib/k13/{models,openvino,qdrant}` and `/opt/k13/qdrant`,
builds `.venv` (portal + RSS sync) and, if needed, `.venv-openvino`, downloads the latest native **Qdrant** and its dashboard,
installs the systemd units, a root-owned model-switch helper (with a sudo rule for that one script), and starts everything.
It is safe to run again at any time. After changing `K13_DATA_DIR` or `MODELS_DIR`, run `./scripts/install.sh --units-only`.

**Python version note (Ubuntu 26.04):** the stock Python may be newer than what the OpenVINO/torch wheels support. If pip prints "no matching
distribution", set `OPENVINO_PYTHON` to the interpreter where OpenVINO already works (it needs `fastapi`, `uvicorn`, `transformers`,
`optimum-intel`, `numpy`: `$OPENVINO_PYTHON -m pip install -r embed/requirements.txt`).

## 3. Get a chat model (GGUF) and start llama.cpp
```bash
./scripts/download-model.sh REPO FILE.gguf --switch      # replace REPO and FILE with real names
```
Examples (check the exact repo and file names on huggingface.co; these are the usual ones, not verified here):
```bash
./scripts/download-model.sh unsloth/Qwen3-8B-GGUF  Qwen3-8B-Q4_K_M.gguf  --switch
./scripts/download-model.sh unsloth/Qwen3-14B-GGUF Qwen3-14B-Q4_K_M.gguf
```
- **Pick `Q4_K_M`** for the best size/quality balance. An 8B model is about 5 GB, a 14B about 9 GB.
- **Fit:** the whole model plus its context must fit in RAM that the GPU can use. If a model is too big for the GPU, lower `LLAMA_GPU_LAYERS`
  in `.env` (for example `28`) so the remaining layers run on the CPU (slower but works), or lower `LLAMA_CTX`.
- **Switching in the portal:** every `.gguf` in `MODELS_DIR` appears in the sidebar; pick one and click *Switch Model* (llama-server restarts, 10 to 90 s).
  From the shell: `sudo k13-llm-switch Model.gguf`, list with `./scripts/k13ctl.sh models`.
- **Qwen3 thinking:** `LLM_THINKING=off` asks Qwen3-style templates to answer directly; the portal also hides any `<think>` text.

## 4. Check it
```bash
./scripts/k13ctl.sh status          # all services + health (embed shows NPU or CPU)
./scripts/k13ctl.sh test            # one embedding + one chat reply
```
- Portal: `http://<server-ip>:8501` - Qdrant dashboard: `http://<server-ip>:6333/dashboard`
- The first embedding-server start downloads and exports the BGE model and compiles it for the NPU (a few minutes). Watch: `./scripts/k13ctl.sh logs embed`.
- The RSS sync first runs 2 minutes after boot, then every 5 minutes: `./scripts/k13ctl.sh logs rss`.

## 5. Backups
```bash
sudo apt install -y samba          # only if you want to share /srv/dropit on the network (see the earlier Samba steps)
./scripts/backup_k13.sh --keep 14          # project + Qdrant snapshot
./scripts/backup_postgres.sh --keep 14     # fetch42 + freshrss databases
./scripts/install-cron.sh                  # daily 04:00 and 04:15; --time / --keep / --remove
```
GGUF models and the vector data are not in the project backup (models are re-downloadable; Qdrant is covered by its snapshot).

## 6. Everyday commands
| Task | Command |
|---|---|
| Status / test | `./scripts/k13ctl.sh status` / `test` |
| Restart one service | `./scripts/k13ctl.sh restart llama` (`qdrant`, `embed`, `llama`, `web`, `rss`, `all`) |
| Logs | `./scripts/k13ctl.sh logs llama` (or `journalctl -fu k13-llama`) |
| After editing `.env` | `./scripts/k13ctl.sh restart all` |
| Update code | `./scripts/update.sh` |
| Upgrade Qdrant | `./scripts/install.sh --upgrade-qdrant` |
| Remove services (data kept) | `./scripts/uninstall.sh` |

## 7. Troubleshooting
- **`k13-llama` inactive, "No model selected":** download a model (step 3). Exit code 78 is deliberate so systemd does not restart-loop.
- **llama-server won't start:** `./scripts/k13ctl.sh logs llama`. Wrong binary path -> `LLAMA_SERVER_BIN`; SYCL needs `LLAMA_PRE_SCRIPT`;
  "out of memory" -> lower `LLAMA_GPU_LAYERS` or `LLAMA_CTX`; unknown flag -> your build is old, update llama.cpp or clear `LLAMA_EXTRA_ARGS`.
  The unit passes `--jinja` (needed for chat templates and the thinking switch), which any recent build supports.
- **Embeddings on CPU instead of NPU:** `ls -l /dev/accel/accel0` must be group `render` and your user in it; check the NPU driver and
  `./scripts/k13ctl.sh logs embed` for the compile error. CPU works but is slower.
- **Portal says llama-server not responding:** it is still loading a large model, or no model is selected. Check `k13ctl.sh status`.
- **Model switch button does nothing / permission error:** run `./scripts/install.sh --units-only` (reinstalls the sudo rule).
- **RSS sync fails to connect to PostgreSQL:** wrong `PG_HOST`/`PG_PORT` in `.env`, or the NAS firewall. Test: `nc -zv $PG_HOST $PG_PORT`.
- **`.env` edits ignored:** services read `.env` at start: `./scripts/k13ctl.sh restart all`.
- **Time wrong / apt "not valid yet":** `sudo timedatectl set-ntp true` and `timedatectl`.
