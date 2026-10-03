"""K13 OpenVINO embedding server (native Ubuntu).

Serves text embeddings from an OpenVINO-compiled BGE model, targeting the Intel NPU with a CPU
fallback. Chat/LLM inference is NOT handled here: that is llama.cpp's llama-server.

Endpoints (Ollama- and OpenAI-compatible so existing clients keep working):
  GET  /healthz, /api/tags, /api/version, /v1/models
  POST /api/embeddings, /api/embed, /v1/embeddings
  POST /api/switch-embed-model, GET /api/switch-embed-model/status, POST /api/reload
"""
import os
import time
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("k13-embed")

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
OV_MODELS_DIR = os.getenv("OV_MODELS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CACHE_DIR = os.getenv("OV_CACHE_DIR", os.path.join(OV_MODELS_DIR, ".cache"))
os.makedirs(OV_MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

EMBED_DEVICE = os.getenv("EMBED_DEVICE", "NPU")
DEFAULT_EMBED_REPO = os.getenv("EMBED_MODEL") or "BAAI/bge-small-en-v1.5"
HF_TOKEN = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip() or None

embed_lock = threading.Lock()          # one inference at a time on the NPU
embed_model = None
embed_tokenizer = None
active_embed_device = "Unknown"
embed_init_error = None
current_embed_repo = DEFAULT_EMBED_REPO
embed_switch_status: Dict[str, Any] = {"state": "idle", "model": DEFAULT_EMBED_REPO, "error": None}
active_models_metadata: Dict[str, Any] = {}


# ------------------------------------------------------------------------------
# OpenVINO helpers
# ------------------------------------------------------------------------------
_device_info_cache = None
_device_info_lock = threading.Lock()


def get_device_info():
    """Return (available_devices, {device: full name}); computed once and cached."""
    global _device_info_cache
    with _device_info_lock:
        if _device_info_cache is None:
            devices, details = [], {}
            try:
                import openvino as ov
                core = ov.Core()
                devices = list(core.available_devices)
                for d in devices:
                    try:
                        details[d] = core.get_property(d, "FULL_DEVICE_NAME")
                    except Exception:
                        details[d] = d
            except Exception as e:
                logger.error(f"OpenVINO device discovery failed: {e}")
            _device_info_cache = (devices, details)
        return _device_info_cache


def load_embed_model(repo_id: Optional[str] = None):
    """Load/export the embedding model; NPU first (static shape), CPU fallback."""
    global embed_model, embed_tokenizer, active_embed_device, embed_init_error, current_embed_repo
    import openvino as ov
    from transformers import AutoTokenizer

    target_repo = repo_id or DEFAULT_EMBED_REPO
    base_dir = os.path.join(OV_MODELS_DIR, target_repo.split("/")[-1].lower())
    model_path = os.path.join(base_dir, "openvino_model.xml")
    logger.info(f"Loading embedding model {target_repo} (target {EMBED_DEVICE})...")

    # 1. Tokenizer (local copy first, else Hugging Face)
    try:
        if os.path.exists(os.path.join(base_dir, "tokenizer_config.json")):
            embed_tokenizer = AutoTokenizer.from_pretrained(base_dir)
        else:
            embed_tokenizer = AutoTokenizer.from_pretrained(target_repo, token=HF_TOKEN)
    except Exception as e:
        logger.error(f"Tokenizer load failed: {e}")
        embed_init_error = f"Tokenizer failed: {e}"
        return

    # 2. Export to OpenVINO IR on first use
    if not os.path.exists(model_path):
        logger.info(f"Exporting {target_repo} to OpenVINO IR (first run only)...")
        try:
            from optimum.intel.openvino import OVModelForFeatureExtraction
            base_model = OVModelForFeatureExtraction.from_pretrained(target_repo, export=True, device="CPU", token=HF_TOKEN)
            base_model.save_pretrained(base_dir)
            embed_tokenizer.save_pretrained(base_dir)
        except Exception as e:
            logger.error(f"Export failed: {e}")
            embed_init_error = f"Export failed: {e}"
            return

    core = ov.Core()

    # 3. NPU needs static shapes
    if EMBED_DEVICE == "NPU":
        try:
            model = core.read_model(model_path)
            shapes = {"input_ids": [1, 512], "attention_mask": [1, 512]}
            for inp in model.inputs:
                shapes.setdefault(inp.get_any_name(), [1, 512])
            model.reshape(shapes)
            npu_cache = os.path.join(CACHE_DIR, "npu")
            os.makedirs(npu_cache, exist_ok=True)
            core.set_property("NPU", {"CACHE_DIR": npu_cache})
            embed_model = core.compile_model(model, device_name="NPU")
            active_embed_device, current_embed_repo, embed_init_error = "NPU", target_repo, None
            logger.info("[OK] Embedding model compiled on Intel NPU.")
            return
        except Exception as e:
            logger.error(f"NPU compilation failed: {e}")
            embed_init_error = f"NPU compilation failed: {e}"
            logger.warning("Falling back to CPU for embeddings.")

    # 4. CPU fallback
    try:
        cpu_cache = os.path.join(CACHE_DIR, "cpu")
        os.makedirs(cpu_cache, exist_ok=True)
        core.set_property("CPU", {"CACHE_DIR": cpu_cache})
        embed_model = core.compile_model(core.read_model(model_path), device_name="CPU")
        active_embed_device, current_embed_repo, embed_init_error = "CPU", target_repo, None
        logger.info("[OK] Embedding model compiled on CPU fallback.")
    except Exception as e:
        logger.error(f"CPU fallback failed: {e}")
        embed_init_error = f"CPU fallback failed: {e}"


def _update_metadata():
    short = current_embed_repo.split("/")[-1]
    entry = {"name": short, "model": short, "details": {"family": "bert", "format": "openvino",
                                                         "device": active_embed_device}}
    for alias in {short, current_embed_repo, "nomic-embed-text", "bge-small-en-v1.5", "bge-base-en-v1.5"}:
        active_models_metadata[alias] = entry


def _compute_embedding(text: str) -> List[float]:
    global embed_init_error
    if embed_model is None or embed_tokenizer is None:
        try:
            load_embed_model()
        except Exception as e:
            embed_init_error = f"On-demand initialization error: {e}"
    if embed_model is None or embed_tokenizer is None:
        raise HTTPException(status_code=503, detail=f"Embedding model not initialized: {embed_init_error}")

    pad_max = active_embed_device == "NPU"
    import openvino as ov
    if isinstance(embed_model, ov.CompiledModel):
        inputs = embed_tokenizer(text, padding="max_length" if pad_max else True, truncation=True,
                                 max_length=512, return_tensors="np")
        names = [i.get_any_name() for i in embed_model.inputs]
        results = embed_model({k: v for k, v in inputs.items() if k in names})
        hidden = list(results.values())[0]
        mask = np.expand_dims(inputs["attention_mask"], axis=-1)
        emb = np.sum(hidden * mask, axis=1) / np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return (emb / np.clip(norm, a_min=1e-9, a_max=None))[0].tolist()

    import torch  # only reached for non-compiled (optimum) models
    inputs = embed_tokenizer(text, padding="max_length" if pad_max else True, truncation=True,
                             max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = embed_model(**inputs)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0]
        mask = inputs["attention_mask"].unsqueeze(-1)
        emb = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return torch.nn.functional.normalize(emb, p=2, dim=1)[0].tolist()


def compute_embedding(text: str) -> List[float]:
    """Blocking embedding (serialized). Call via asyncio.to_thread() from async handlers."""
    with embed_lock:
        return _compute_embedding(text)


# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting K13 OpenVINO embedding server...")
    get_device_info()
    try:
        load_embed_model()
    except Exception as e:
        logger.error(f"Initial embedding load failed: {e}")
    _update_metadata()
    yield
    logger.info("Shutting down.")


app = FastAPI(title="K13 OpenVINO Embedding Server", version="2.0.0", lifespan=lifespan)


class EmbedRequest(BaseModel):
    model: Optional[str] = "nomic-embed-text"
    prompt: Optional[str] = None
    input: Optional[Union[str, List[str]]] = None


class OpenAIEmbeddingRequest(BaseModel):
    model: str = "nomic-embed-text"
    input: Union[str, List[str]]


class SwitchEmbedModelRequest(BaseModel):
    model: str = Field(..., description="Hugging Face repo, e.g. 'BAAI/bge-base-en-v1.5'")


@app.get("/")
@app.get("/healthz")
async def health():
    devices, details = await asyncio.to_thread(get_device_info)
    loaded = embed_model is not None
    return {
        "status": "healthy" if loaded else "unhealthy",
        "role": "embeddings",
        "embed_hardware": active_embed_device,
        "embed_loaded": loaded,
        "embed_model": current_embed_repo,
        "embed_error": embed_init_error,
        "available_devices": devices,
        "device_details": details,
        "model_switch": embed_switch_status,
    }


@app.get("/api/tags")
@app.get("/api/models")
@app.get("/v1/models")
async def models():
    return {"models": list(active_models_metadata.values()),
            "data": [{"id": k, "object": "model"} for k in active_models_metadata]}


@app.get("/api/version")
async def version():
    return {"version": "2.0.0-openvino-embed"}


@app.post("/api/reload")
async def reload_models():
    await asyncio.to_thread(load_embed_model)
    _update_metadata()
    return {"embed_hardware": active_embed_device, "embed_loaded": embed_model is not None, "error": embed_init_error}


@app.post("/api/embeddings")
async def embeddings(req: EmbedRequest):
    query = req.prompt or (req.input if isinstance(req.input, str) else (req.input[0] if req.input else ""))
    if not query:
        raise HTTPException(status_code=400, detail="Missing prompt or input text.")
    return {"embedding": await asyncio.to_thread(compute_embedding, query)}


@app.post("/api/embed")
async def embed_batch(req: EmbedRequest):
    inputs = req.input or ([req.prompt] if req.prompt else [])
    if isinstance(inputs, str):
        inputs = [inputs]
    return {"embeddings": await asyncio.to_thread(lambda: [compute_embedding(t) for t in inputs])}


@app.post("/v1/embeddings")
async def openai_embeddings(req: OpenAIEmbeddingRequest):
    inputs = [req.input] if isinstance(req.input, str) else req.input
    data = []
    for idx, text in enumerate(inputs):
        data.append({"object": "embedding", "index": idx, "embedding": await asyncio.to_thread(compute_embedding, text)})
    return {"object": "list", "data": data, "model": req.model, "usage": {"prompt_tokens": 0, "total_tokens": 0}}


@app.post("/api/switch-embed-model")
async def switch_embed_model(req: SwitchEmbedModelRequest):
    """Hot-swap the embedding model. A different vector size needs the Qdrant collection re-indexed."""
    global embed_model
    repo = req.model.strip()
    if not repo:
        raise HTTPException(status_code=400, detail="'model' field is required.")
    if embed_switch_status["state"] == "loading":
        return JSONResponse(status_code=409, content={"status": "busy", "model": embed_switch_status["model"]})
    embed_switch_status.update(state="loading", model=repo, error=None)

    def _do_switch():
        global embed_model
        previous = current_embed_repo
        with embed_lock:
            embed_model = None
            import gc
            gc.collect()
            time.sleep(1)
            load_embed_model(repo)
            if embed_model is None and previous != repo:      # roll back
                load_embed_model(previous)
                embed_switch_status["error"] = f"Could not load {repo}: {embed_init_error}. Restored {previous}."
                embed_switch_status["state"] = "error"
                return
        _update_metadata()
        if embed_model is not None:
            embed_switch_status["state"] = "ready"
        else:
            embed_switch_status.update(state="error", error=embed_init_error or "Unknown error")

    threading.Thread(target=_do_switch, daemon=True).start()
    return {"status": "loading", "model": repo}


@app.get("/api/switch-embed-model/status")
async def switch_embed_status():
    return {"switch_status": embed_switch_status, "current_model": current_embed_repo,
            "embed_loaded": embed_model is not None, "embed_hardware": active_embed_device}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("EMBED_HOST", "127.0.0.1")
    port = int(os.getenv("EMBED_PORT", "11434"))
    logger.info(f"Listening on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
