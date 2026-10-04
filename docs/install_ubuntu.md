# K13 AI Stack - Install Guide (Ubuntu Server, native OpenVINO)

You already have **OpenVINO** installed on this machine. This guide wires everything else around it. There are no containers and no llama.cpp:
one OpenVINO server runs the chat LLM on the GPU and the embeddings on the NPU.

## 0. Check what you have
```bash
ls -l /dev/accel /dev/dri                                  # NPU (accel0) and GPU (renderD128)
python3 -c "import openvino as ov; print(ov.Core().available_devices)"      # expect ['CPU','GPU','NPU']
python3 -c "import openvino_genai; print('genai ok')"                       # needed for the LLM
```
- If OpenVINO lives in a venv, note that Python's path (for example `/home/e/ov/bin/python`): it goes in `.env` as `OPENVINO_PYTHON`.
  That Python also needs `openvino-genai`, `fastapi`, `uvicorn`, `transformers`, `optimum-intel`, `torch`, `huggingface-hub` (the installer tells you what is missing).
- If OpenVINO came from an archive that needs `source .../setupvars.sh`, put that script's path in `OPENVINO_SETUPVARS`.
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
nano .env
./scripts/install.sh        # second run installs everything
```
Set at least:

| Setting | Meaning |
|---|---|
| `OPENVINO_PYTHON` / `OPENVINO_SETUPVARS` | only if OpenVINO is not in this project's own venv (blank = the installer creates `.venv-openvino`) |
| `DEFAULT_MODEL` | OpenVINO LLM repo on Hugging Face (default `OpenVINO/Qwen3-8B-int4-ov`) |
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` (768 dims) is the default and `install.sh` resets it on every run (`--keep-embed-model` to opt out). If the dimension changes, the RSS sync recreates the Qdrant collection and re-embeds everything from Postgres automatically |
| `FRESHRSS_*`, `PG_*` | your NAS FreshRSS and PostgreSQL logins |
| `HF_TOKEN` | Hugging Face token (model downloads) |
| `DROPIT_DIR` | backup folder (default `/srv/dropit`) |

What the installer does: installs OS packages, creates `/var/lib/k13/{openvino,qdrant}` and `/opt/k13/qdrant`, builds `.venv` (portal + RSS sync) and,
if needed, `.venv-openvino`, downloads the latest native **Qdrant** and its dashboard, installs the systemd units and starts everything.
It is safe to run again at any time.

**Python version note (Ubuntu 26.04):** the stock Python may be newer than what the OpenVINO/torch wheels support. If pip prints "no matching
distribution", set `OPENVINO_PYTHON` to the interpreter where OpenVINO already works and run
`$OPENVINO_PYTHON -m pip install -r openvino/requirements.txt` for whatever is missing.

## 3. First start and models
The first start of `k13-openvino` downloads the LLM (about 5.5 GB for Qwen3-8B) and exports and compiles the embedding model for the NPU. It can take
10 to 20 minutes:
```bash
./scripts/k13ctl.sh logs openvino        # Ctrl+C to leave; the service keeps running
./scripts/k13ctl.sh status               # LLM and embeddings should show GPU and NPU
./scripts/k13ctl.sh test                 # one embedding + one chat reply
```
- **Portal:** `http://<server-ip>:8501` - **Qdrant dashboard:** `http://<server-ip>:6333/dashboard`
- **Switch the LLM** in the portal sidebar (it downloads the model if needed). If a model does not fit on the GPU the switch is rolled back and the previous
  model stays loaded. On this hardware the GPU runtime caps at about **6.8 GB**, so Qwen3-8B int4 is the largest that fits. Smaller models: Phi-4 Mini, Llama 3.2 3B, Qwen3 4B.
- **Check the GPU's limit yourself:**
  ```bash
  python3 -c "import openvino as ov; print(round(ov.Core().get_property('GPU','GPU_DEVICE_TOTAL_MEM_SIZE')/1024**3,1),'GB')"
  ```
- The RSS sync first runs 2 minutes after boot, then every 5 minutes: `./scripts/k13ctl.sh logs rss`.

## 4. Backups and the network share
`install.sh` installs the backup scripts but does **not** schedule them or create a share unless you ask. One command does both:
```bash
./scripts/install.sh --with-share --with-backups
```
Or step by step:
```bash
./scripts/setup-share.sh                   # creates /srv/dropit and shares it over SMB (asks for a share password once)
./scripts/backup_k13.sh --keep 14          # test: project + Qdrant snapshot -> DROPIT_DIR
./scripts/backup_postgres.sh --keep 14     # test: fetch42 + freshrss databases
./scripts/install-cron.sh                  # daily 04:00 and 04:15; --time / --keep / --remove
```
- **The share:** only your user can sign in, only from your own LAN (auto-detected; override with `--subnet 192.168.4.0/24`), SMB2+ only, no guests.
  Open `\\<server-ip>\dropit` on Windows or `smb://<server-ip>/dropit` on a Mac. The password is separate from your Linux login: `sudo smbpasswd e` changes it.
  Remove the share any time with `./scripts/setup-share.sh --remove` (files are kept).
- **If `/srv/dropit` is a mounted NAS share** the script stops: unmount it and remove its `/etc/fstab` line first.
- **Backups live on this server's disk**, so also copy them elsewhere now and then.
- Downloaded models and the vector data folder are not in the project backup (models re-download; Qdrant is covered by its snapshot).

## 5. Everyday commands
| Task | Command |
|---|---|
| Status / test | `./scripts/k13ctl.sh status` / `test` |
| Restart one service | `./scripts/k13ctl.sh restart openvino` (`qdrant`, `openvino`, `web`, `rss`, `all`) |
| Logs | `./scripts/k13ctl.sh logs openvino` (or `journalctl -fu k13-openvino`) |
| After editing `.env` | `./scripts/k13ctl.sh restart all` |
| Update code | `./scripts/update.sh` |
| Upgrade Qdrant | `./scripts/install.sh --upgrade-qdrant` |
| Remove services (data kept) | `./scripts/uninstall.sh` |

## 6. Troubleshooting
- **`k13-openvino` keeps restarting:** read `./scripts/k13ctl.sh logs openvino`. Missing Python packages -> `$OPENVINO_PYTHON -m pip install -r openvino/requirements.txt`.
- **LLM on CPU instead of GPU / NPU embeddings on CPU:** check `ls -l /dev/dri /dev/accel` (group `render`), that the Intel GPU/NPU drivers are installed, and the compile error in the logs.
- **"Switch failed ... did not fit on the Intel GPU":** the model is too big for the ~6.8 GB GPU budget; the previous model was restored. Choose a smaller one.
- **Portal says inference server unreachable:** the OpenVINO service may still be loading models; check `k13ctl.sh status`.
- **RSS sync fails to connect to PostgreSQL:** wrong `PG_HOST`/`PG_PORT` in `.env`, or the NAS firewall. Test: `nc -zv $PG_HOST $PG_PORT`.
- **`.env` edits ignored:** services read `.env` at start: `./scripts/k13ctl.sh restart all`.
- **Time wrong / apt "not valid yet":** `sudo timedatectl set-ntp true`.
- **`apt update` fails on a third-party repo:** disable that source in `/etc/apt/sources.list.d/`; the installer continues anyway.
