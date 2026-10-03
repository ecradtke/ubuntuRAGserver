# Platform Rule: K13 AI Hub on Ubuntu Server (native OpenVINO, no containers)

- Hardware: Intel Core Ultra (Lunar Lake class) with Intel GPU + NPU. OS: Ubuntu Server with a minimal desktop. Services run under **systemd**.
- All inference is **OpenVINO** (`openvino/server.py`, port 11434): LLM on the GPU, embeddings (bge-base, 768 dims, default) on the NPU, CPU fallback. Vector DB: native **Qdrant** on 6333.
- No Docker, no containers, no Ollama, no llama.cpp. Use `scripts/k13ctl.sh`, `scripts/install.sh`, `scripts/update.sh`.
- The GPU runtime caps at ~6.8 GB here: Qwen3-8B int4 is the largest LLM that fits.
