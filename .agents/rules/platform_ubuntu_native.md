# Platform Rule: K13 AI Hub on Ubuntu Server (native, no containers)

- Hardware: Intel Core Ultra (Lunar Lake class) with Intel GPU + NPU. OS: Ubuntu Server with a minimal desktop. Services run under **systemd**.
- Chat LLM: **llama.cpp `llama-server`** (GGUF) on port 8080. Embeddings: **OpenVINO** server on port 11434 (NPU, CPU fallback). Vector DB: native **Qdrant** on 6333.
- No Docker, no containers, no Ollama. Use `scripts/k13ctl.sh`, `scripts/install.sh`, `scripts/update.sh`.
- Embedding dimension must match the Qdrant collection (384 for bge-small, 768 for bge-base).
