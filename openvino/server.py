import os
import sys
import asyncio
import time
import json
import queue
import threading
import logging
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import torch
import openvino as ov
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("openvino-server")

# ------------------------------------------------------------------------------
# Configuration & Hardware Devices
# ------------------------------------------------------------------------------
MODELS_DIR = os.getenv("MODELS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CACHE_DIR = os.getenv("OV_CACHE_DIR", os.path.join(MODELS_DIR, ".cache"))
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

INFERENCE_DEVICE = os.getenv("LLM_DEVICE") or os.getenv("INFERENCE_DEVICE", "GPU")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "NPU")
DEFAULT_LLM_REPO = os.getenv("DEFAULT_MODEL") or os.getenv("DEFAULT_MODEL_REPO", "OpenVINO/Phi-3.5-mini-instruct-int4-ov")
DEFAULT_EMBED_REPO = os.getenv("EMBED_MODEL") or os.getenv("DEFAULT_EMBED_REPO", "BAAI/bge-small-en-v1.5")
# Reasoning models (Qwen3, ...) burn their token budget on hidden <think> reasoning and can run out
# before writing an answer. Thinking is OFF by default; set LLM_THINKING=on to allow it.
# A model switch that only fits on the CPU is treated as a failure (and rolled back) unless this is on.
SWITCH_ALLOW_CPU = os.getenv("SWITCH_ALLOW_CPU", "off").strip().lower() in ("on", "1", "true", "yes")
LLM_THINKING = os.getenv("LLM_THINKING", "off").strip().lower() in ("on", "1", "true", "yes")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None
if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip() or None

# One inference at a time per device (the pipelines are not thread-safe), but the GPU (LLM)
# and NPU (embeddings) locks are independent so both devices can work at the same time.
llm_lock = threading.Lock()
embed_lock = threading.Lock()

llm_pipeline = None
llm_tokenizer = None
embed_model = None
embed_tokenizer = None
active_models_metadata = {}
active_llm_device = "Unknown"
active_embed_device = "Unknown"
llm_init_error = None
embed_init_error = None
current_llm_repo = DEFAULT_LLM_REPO
current_embed_repo = DEFAULT_EMBED_REPO
model_switch_status: Dict[str, Any] = {"state": "idle", "model": DEFAULT_LLM_REPO, "error": None}
embed_switch_status: Dict[str, Any] = {"state": "idle", "model": DEFAULT_EMBED_REPO, "error": None}

# ------------------------------------------------------------------------------
# OpenVINO Runtime & Models Initialization
# ------------------------------------------------------------------------------
def check_inference_devices():
    """Identifies optimal Intel hardware targets: GPU for LLM, NPU for Embeddings."""
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        logger.info(f"OpenVINO detected available devices: {devices}")
        
        # LLM target: Intel GPU prioritized per user request
        req_llm = os.getenv("LLM_DEVICE") or os.getenv("INFERENCE_DEVICE", "GPU")
        matching_gpu = next((d for d in devices if d == req_llm or (req_llm.startswith("GPU") and d.startswith("GPU"))), None)
        if matching_gpu:
            llm_dev = matching_gpu
            logger.info(f"[✓] Intel {llm_dev} selected for LLM inference.")
        elif any(d.startswith("GPU") for d in devices):
            llm_dev = next(d for d in devices if d.startswith("GPU"))
            logger.info(f"[✓] Intel {llm_dev} selected for LLM inference.")
        else:
            llm_dev = req_llm
            logger.info(f"[✓] Intel {llm_dev} targeted for LLM inference.")

        # Embedding target: Intel NPU prioritized per user request
        req_embed = os.getenv("EMBED_DEVICE", "NPU")
        matching_npu = next((d for d in devices if d == req_embed or (req_embed.startswith("NPU") and d.startswith("NPU"))), None)
        if matching_npu:
            embed_dev = matching_npu
            logger.info(f"[✓] Intel {embed_dev} selected for embedding generation.")
        elif any(d.startswith("NPU") for d in devices):
            embed_dev = next(d for d in devices if d.startswith("NPU"))
            logger.info(f"[✓] Intel {embed_dev} selected for embedding generation.")
        else:
            embed_dev = req_embed
            logger.info(f"[✓] Intel {embed_dev} targeted for embedding generation.")

        return llm_dev, embed_dev
    except Exception as e:
        logger.error(f"Error checking OpenVINO devices: {e}")
        return "GPU", "NPU"


def ensure_llm_downloaded(target_repo: str):
    """Download an LLM repo from Hugging Face if it is not already on disk (raises on failure)."""
    from huggingface_hub import snapshot_download
    model_local_dir = os.path.join(MODELS_DIR, target_repo.split("/")[-1].lower())
    if os.path.exists(os.path.join(model_local_dir, "openvino_model.xml")):
        return
    logger.info(f"Downloading LLM model from Hugging Face: {target_repo}...")
    snapshot_download(
        repo_id=target_repo,
        local_dir=model_local_dir,
        token=HF_TOKEN,
        allow_patterns=["*.xml", "*.bin", "*.json", "*.txt", "*.model", "*.py", "*.jinja"]
    )


def load_llm_pipeline(device: str = "GPU", repo_id: str = None):
    global llm_pipeline, llm_tokenizer, active_llm_device, llm_init_error, current_llm_repo
    import openvino_genai as ov_genai
    from huggingface_hub import snapshot_download

    target_repo = repo_id or DEFAULT_LLM_REPO
    repo_name = target_repo.split("/")[-1]
    model_local_dir = os.path.join(MODELS_DIR, repo_name.lower())
    model_path = model_local_dir

    # Download if not already present
    if not os.path.exists(os.path.join(model_local_dir, "openvino_model.xml")):
        logger.info(f"Downloading LLM model from Hugging Face: {target_repo}...")
        try:
            snapshot_download(
                repo_id=target_repo,
                local_dir=model_local_dir,
                token=HF_TOKEN,
                allow_patterns=["*.xml", "*.bin", "*.json", "*.txt", "*.model", "*.py", "*.jinja"]
            )
        except Exception as de:
            logger.error(f"Failed to download LLM model {target_repo}: {de}")
            llm_init_error = f"Download error: {de}"
            return

    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(model_local_dir, trust_remote_code=True)
        logger.info(f"[✓] Tokenizer loaded successfully for {target_repo}.")
    except Exception as te:
        logger.warning(f"Could not load AutoTokenizer from {model_local_dir}: {te}")

    # Determine target device — env var takes precedence over the argument.
    target_device = os.getenv("LLM_DEVICE") or os.getenv("INFERENCE_DEVICE") or device or "GPU"

    logger.info(f"Compiling LLM pipeline [{target_repo}] on Intel {target_device}...")
    try:
        # No extra plugin properties: KV_CACHE_PRECISION/ENABLE_MMAP/CACHE_DIR overrides broke GPU execution.
        llm_pipeline = ov_genai.LLMPipeline(model_path, target_device)
        active_llm_device = target_device
        current_llm_repo = target_repo
        llm_init_error = None
        logger.info(f"[✓] LLM pipeline [{target_repo}] loaded on Intel {target_device}.")
        return
    except Exception as e:
        logger.error(f"Failed to load LLM [{target_repo}] on Intel {target_device}: {e}")
        llm_init_error = f"{target_device} compilation failed: {e}"
        if target_device != "CPU":
            logger.warning("Attempting fallback compilation to Intel CPU (OpenVINO accelerated)...")
            try:
                llm_pipeline = ov_genai.LLMPipeline(model_path, "CPU")
                active_llm_device = "CPU"
                current_llm_repo = target_repo
                llm_init_error = None
                logger.info(f"[✓] LLM pipeline [{target_repo}] compiled on CPU fallback.")
                return
            except Exception as ce:
                logger.error(f"CPU fallback failed: {ce}")
                llm_init_error = f"CPU fallback failed: {ce}"


def load_embed_model(device: str = "NPU", repo_id: str = None):
    global embed_model, embed_tokenizer, active_embed_device, embed_init_error, current_embed_repo
    import openvino as ov
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download

    target_repo = repo_id or DEFAULT_EMBED_REPO
    repo_name = target_repo.split("/")[-1].lower()
    embed_base_dir = os.path.join(MODELS_DIR, repo_name)
    model_path = os.path.join(embed_base_dir, "openvino_model.xml")

    # Embedding ALWAYS targets NPU — env var override is respected but the
    # argument 'device' passed by the caller is ignored in favour of the env.
    # This prevents a runtime race where check_inference_devices() might return
    # a different device if the NPU driver enumeration is slow at startup.
    target_embed_device = os.getenv("EMBED_DEVICE", "NPU")
    logger.info(f"Loading embedding model ({target_repo}) targeting {target_embed_device}...")

    # 1. Safely load tokenizer
    try:
        if os.path.exists(os.path.join(embed_base_dir, "tokenizer_config.json")):
            embed_tokenizer = AutoTokenizer.from_pretrained(embed_base_dir)
            logger.info(f"[✓] Embedding tokenizer loaded from local cache: {embed_base_dir}")
        else:
            embed_tokenizer = AutoTokenizer.from_pretrained(target_repo, token=HF_TOKEN)
            logger.info(f"[✓] Embedding tokenizer loaded from {target_repo}")
    except Exception as te:
        logger.warning(f"Could not load tokenizer from local cache: {te}. Retrying with remote repo...")
        try:
            embed_tokenizer = AutoTokenizer.from_pretrained(target_repo, token=HF_TOKEN)
        except Exception as te2:
            logger.error(f"Fatal error loading tokenizer: {te2}")
            embed_init_error = f"Tokenizer failed: {te2}"

    # 2. Ensure base OpenVINO IR exists on disk (export via optimum-intel if needed)
    if not os.path.exists(model_path):
        logger.info(f"Exporting embedding model ({target_repo}) to OpenVINO IR...")
        try:
            from optimum.intel.openvino import OVModelForFeatureExtraction
            base_model = OVModelForFeatureExtraction.from_pretrained(
                target_repo,
                export=True,
                device="CPU",
                token=HF_TOKEN
            )
            base_model.save_pretrained(embed_base_dir)
            if embed_tokenizer:
                embed_tokenizer.save_pretrained(embed_base_dir)
            logger.info(f"[✓] Embedding IR saved to {embed_base_dir}")
        except Exception as ee:
            logger.error(f"Failed to export embedding model: {ee}")
            embed_init_error = f"Export failed: {ee}"
            return

    core = ov.Core()

    # 3. Always attempt NPU first with static shape (required by NPU driver).
    # The device is NOT taken from the caller argument to avoid accidental
    # CPU routing when the NPU driver is still initialising at startup.
    if target_embed_device == "NPU":
        try:
            logger.info("Reading model IR and applying static shape for Intel NPU...")
            model = core.read_model(model_path)

            shapes = {"input_ids": [1, 512], "attention_mask": [1, 512]}
            for inp in model.inputs:
                name = inp.get_any_name()
                if name not in shapes:
                    shapes[name] = [1, 512]
            model.reshape(shapes)
            logger.info(f"[✓] Model reshaped to static shape: {shapes}")

            npu_cache = os.path.join(CACHE_DIR, "npu")
            os.makedirs(npu_cache, exist_ok=True)
            core.set_property("NPU", {"CACHE_DIR": npu_cache})

            logger.info("Compiling model for device: NPU...")
            compiled_embed_model = core.compile_model(model, device_name="NPU")
            embed_model = compiled_embed_model
            active_embed_device = "NPU"
            current_embed_repo = target_repo
            embed_init_error = None
            logger.info("[✓] Embedding model compiled and ready on Intel NPU.")
            return
        except Exception as ne:
            logger.error(f"Failed to compile embedding model on NPU: {ne}")
            embed_init_error = f"NPU compilation failed: {ne}"
            logger.warning("NPU unavailable — falling back to CPU for embeddings.")

    # 4. Fallback: CPU only (never GPU for embeddings — NPU is the target).
    logger.warning("Compiling embedding model on Intel CPU (OpenVINO accelerated fallback)...")
    try:
        model_cpu = core.read_model(model_path)
        cpu_cache = os.path.join(CACHE_DIR, "cpu")
        os.makedirs(cpu_cache, exist_ok=True)
        core.set_property("CPU", {"CACHE_DIR": cpu_cache})
        embed_model = core.compile_model(model_cpu, device_name="CPU")
        active_embed_device = "CPU"
        current_embed_repo = target_repo
        embed_init_error = None
        logger.info("[✓] Embedding model compiled and ready on Intel CPU fallback.")
        return
    except Exception as ce:
        logger.error(f"CPU fallback compilation failed: {ce}")
        embed_init_error = f"CPU fallback failed: {ce}"


def _update_active_model_metadata():
    """Refresh active_models_metadata to reflect the currently loaded LLM repo."""
    model_short_name = current_llm_repo.split("/")[-1]
    is_phi = "phi" in model_short_name.lower()
    is_medium = "medium" in model_short_name.lower()
    primary_model_tag = f"OpenVINO-{model_short_name}"

    entry = {
        "name": primary_model_tag,
        "model": primary_model_tag,
        "details": {
            "family": "phi" if is_phi else "llama",
            "parameter_size": "14B" if is_medium else ("3.8B" if is_phi else "3B"),
            "quantization_level": "INT4",
            "format": "openvino"
        }
    }
    active_models_metadata[primary_model_tag] = entry
    # Backward-compat aliases so existing callers keep working
    for _alias in ["llama3.2:latest", "llama3.2:3b", "llama3.2", "phi3.5:latest", "phi3.5"]:
        active_models_metadata[_alias] = entry
    active_models_metadata[model_short_name] = entry
    active_models_metadata[current_llm_repo] = entry


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing K13 OpenVINO Intel AI Server...")
    llm_dev, embed_dev = check_inference_devices()

    # 1. Load embedding model FIRST on NPU (takes ~3s, unblocks RSS pipeline immediately)
    try:
        load_embed_model(embed_dev)
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")

    # 2. Load LLM pipeline on GPU / CPU fallback (takes ~3s on CPU)
    try:
        load_llm_pipeline(llm_dev)
    except Exception as e:
        logger.error(f"Error loading LLM: {e}")

    # --- Device mismatch sanity check ---
    if active_llm_device not in ("GPU", llm_dev):
        logger.warning(
            f"\u26a0\ufe0f  LLM is running on [{active_llm_device}], NOT [{llm_dev}] as configured! "
            f"GPU compilation likely failed. Check that your user is in the 'render' group and the Intel GPU driver is installed (ls -l /dev/dri). "
            f"LLM error: {llm_init_error}"
        )
    else:
        logger.info(f"[\u2713] LLM device confirmed: {active_llm_device}")

    if active_embed_device not in ("NPU", embed_dev):
        logger.warning(
            f"\u26a0\ufe0f  Embeddings running on [{active_embed_device}], NOT [{embed_dev}] as configured! "
            f"NPU compilation likely failed. Check /dev/accel/accel0 (group 'render') and the Intel NPU driver. "
            f"Embed error: {embed_init_error}"
        )
    else:
        logger.info(f"[\u2713] Embed device confirmed: {active_embed_device}")

    _update_active_model_metadata()

    embed_short = current_embed_repo.split("/")[-1]
    _embed_entry = {
        "name": embed_short,
        "model": embed_short,
        "details": {
            "family": "bert",
            "parameter_size": "base" if "base" in embed_short.lower() else "small",
            "format": "openvino"
        }
    }
    active_models_metadata[embed_short] = _embed_entry
    # Backward-compat aliases
    active_models_metadata["nomic-embed-text"] = _embed_entry
    active_models_metadata["bge-small-en-v1.5"] = _embed_entry
    active_models_metadata["bge-base-en-v1.5"] = _embed_entry
    active_models_metadata[current_embed_repo] = _embed_entry
    yield
    logger.info("Shutting down OpenVINO Intel AI Server...")

app = FastAPI(
    title="K13 OpenVINO NPU Inference Server",
    version="1.0.0",
    lifespan=lifespan
)

# ------------------------------------------------------------------------------
# Pydantic Schemas for Ollama & OpenAI compatibility
# ------------------------------------------------------------------------------
class OllamaChatMessage(BaseModel):
    role: str
    content: str

class OllamaChatRequest(BaseModel):
    model: str = "llama3.2:latest"
    messages: List[OllamaChatMessage]
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None

class OllamaGenerateRequest(BaseModel):
    model: str = "llama3.2:latest"
    prompt: str
    system: Optional[str] = None
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None

class OllamaEmbedRequest(BaseModel):
    model: Optional[str] = "nomic-embed-text"
    prompt: Optional[str] = None
    input: Optional[Union[str, List[str]]] = None

class OpenAIChatCompletionRequest(BaseModel):
    model: str = "llama3.2:latest"
    messages: List[Dict[str, str]]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024

class OpenAIEmbeddingRequest(BaseModel):
    model: str = "nomic-embed-text"
    input: Union[str, List[str]]

class SwitchModelRequest(BaseModel):
    model: str = Field(..., description="Hugging Face repo ID of the OpenVINO LLM model to switch to.")

class SwitchEmbedModelRequest(BaseModel):
    model: str = Field(..., description="Hugging Face repo ID of the embedding model to switch to, e.g. 'BAAI/bge-base-en-v1.5'")

# ------------------------------------------------------------------------------
# Helper Utilities
# ------------------------------------------------------------------------------
def compute_embedding(text: str) -> List[float]:
    """Blocking NPU/CPU embedding. Call via asyncio.to_thread() from async handlers."""
    with embed_lock:
        return _compute_embedding(text)


def locked_generate(prompt, gen_config, streamer=None):
    """Blocking LLM generation on the GPU, serialized by llm_lock."""
    with llm_lock:
        if streamer is None:
            return llm_pipeline.generate(prompt, gen_config)
        return llm_pipeline.generate(prompt, gen_config, streamer)


def _compute_embedding(text: str) -> List[float]:
    global embed_model, embed_tokenizer, active_embed_device, embed_init_error
    if not embed_model or not embed_tokenizer:
        logger.warning("Embedding model not ready. Attempting on-demand initialization...")
        try:
            _, embed_dev = check_inference_devices()
            load_embed_model(embed_dev)
        except Exception as e:
            logger.error(f"On-demand initialization error: {e}")
            embed_init_error = f"On-demand initialization error: {e}"

    if not embed_model or not embed_tokenizer:
        detail_msg = f"Embedding model not initialized: {embed_init_error}" if embed_init_error else "Embedding model not initialized. Check OpenVINO drivers and local model cache."
        raise HTTPException(
            status_code=503,
            detail=detail_msg
        )
    
    pad_max = (active_embed_device == "NPU")
    import openvino as ov
    if isinstance(embed_model, ov.CompiledModel):
        inputs = embed_tokenizer(
            text,
            padding="max_length" if pad_max else True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        input_names = [inp.get_any_name() for inp in embed_model.inputs]
        feed_dict = {k: v for k, v in inputs.items() if k in input_names}
        results = embed_model(feed_dict)
        last_hidden = list(results.values())[0]
        attention_mask = np.expand_dims(inputs["attention_mask"], axis=-1)
        embeddings = np.sum(last_hidden * attention_mask, axis=1) / np.clip(np.sum(attention_mask, axis=1), a_min=1e-9, a_max=None)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_embeddings = (embeddings / np.clip(norm, a_min=1e-9, a_max=None))[0].tolist()
        return norm_embeddings
    else:
        inputs = embed_tokenizer(
            text, 
            padding="max_length" if pad_max else True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = embed_model(**inputs)
            # Sentence embedding directly available in some models
            if hasattr(outputs, "sentence_embedding") and outputs.sentence_embedding is not None:
                norm_embeddings = torch.nn.functional.normalize(outputs.sentence_embedding, p=2, dim=1)
                return norm_embeddings[0].tolist()

            # Mean pooling over last_hidden_state
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                if isinstance(outputs, (list, tuple)):
                    last_hidden = outputs[0]
                elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
                    last_hidden = outputs["last_hidden_state"]

            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = torch.sum(last_hidden * attention_mask, dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return norm_embeddings[0].tolist()


def format_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """Format chat messages according to model chat template or standard fallback."""
    if llm_tokenizer is not None and getattr(llm_tokenizer, "chat_template", None):
        try:
            # enable_thinking=False makes Qwen3-style templates pre-close the <think> block so the
            # model answers directly; templates that don't use the flag simply ignore it.
            return llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=LLM_THINKING
            )
        except Exception as e:
            logger.warning(f"Error applying tokenizer chat_template: {e}")

    # Fallback chat format
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt += f"<|{role}|>\n{content}<|end|>\n"
    prompt += "<|assistant|>\n"
    return prompt

# ------------------------------------------------------------------------------
# Ollama-Compatible Endpoints (/api/*)
# ------------------------------------------------------------------------------
@app.get("/api/tags")
@app.get("/api/models")
async def get_tags():
    """Lists available models in Ollama API format."""
    return {
        "models": list(active_models_metadata.values())
    }

@app.get("/api/version")
async def get_version():
    return {"version": "0.1.0-openvino"}

_device_info_cache = None
_device_info_lock = threading.Lock()


def get_device_info():
    """Return (available_devices, {device: full name}); computed once and cached."""
    global _device_info_cache
    with _device_info_lock:
        if _device_info_cache is None:
            devices, details = [], {}
            try:
                core = ov.Core()
                devices = list(core.available_devices)
                for d in devices:
                    try:
                        details[d] = core.get_property(d, "FULL_DEVICE_NAME")
                    except Exception:
                        details[d] = d
            except Exception:
                pass
            _device_info_cache = (devices, details)
        return _device_info_cache


@app.get("/")
@app.get("/healthz")
async def health():
    is_healthy = (llm_pipeline is not None) and (embed_model is not None)
    status_str = "healthy" if is_healthy else ("degraded" if (llm_pipeline is not None or embed_model is not None) else "unhealthy")
    
    # Device discovery creates an ov.Core() and queries every device, which can take several seconds
    # (and used to freeze the event loop). Do it once, off the loop, then serve it from cache.
    detected_devices, device_details = await asyncio.to_thread(get_device_info)

    return {
        "status": status_str,
        "llm_hardware": active_llm_device,
        "embed_hardware": active_embed_device,
        "llm_loaded": llm_pipeline is not None,
        "embed_loaded": embed_model is not None,
        "available_devices": detected_devices,
        "device_details": device_details,
        "llm_error": llm_init_error,
        "embed_error": embed_init_error,
        "current_model": current_llm_repo,
        "model_switch": model_switch_status,
        "models": list(active_models_metadata.keys())
    }

@app.get("/api/diag")
def diag():
    """Runs a short generation and reports measured throughput; ~1-3 tok/s on an 8B model means CPU."""
    if not llm_pipeline:
        raise HTTPException(status_code=503, detail=f"LLM not ready: {llm_init_error}")
    import openvino_genai as ov_genai
    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = 400
    start = time.time()
    output = llm_pipeline.generate("Count from 1 to 40.", gen_config)
    elapsed = time.time() - start
    return {
        "llm_hardware": active_llm_device,
        "model": current_llm_repo,
        "elapsed_sec": round(elapsed, 2),
        "approx_tokens_per_sec": round(400 / elapsed, 2) if elapsed > 0 else None,
        "output_preview": output[:100],
    }


@app.post("/api/reload")
@app.get("/api/reload")
async def reload_models():
    """Trigger on-demand reload and compilation of LLM and embedding pipelines."""
    llm_dev, embed_dev = check_inference_devices()
    try:
        load_llm_pipeline(llm_dev)
    except Exception as e:
        logger.error(f"Manual reload LLM error: {e}")
    try:
        load_embed_model(embed_dev)
    except Exception as e:
        logger.error(f"Manual reload embedding error: {e}")
    _update_active_model_metadata()
    return await health()


@app.post("/api/switch-model")
async def switch_model(req: SwitchModelRequest):
    """Hot-swap the active LLM to any OpenVINO HF repo without restarting the service."""
    global llm_pipeline, model_switch_status

    requested_repo = req.model.strip()
    if not requested_repo:
        raise HTTPException(status_code=400, detail="'model' field is required.")

    if model_switch_status["state"] == "loading":
        return JSONResponse(
            status_code=409,
            content={
                "status": "busy",
                "message": "A model switch is already in progress.",
                "model": model_switch_status["model"]
            }
        )

    # Kick off the switch in a background thread so this request returns immediately
    model_switch_status["state"] = "loading"
    model_switch_status["model"] = requested_repo
    model_switch_status["error"] = None

    def _do_switch():
        global llm_pipeline, model_switch_status
        previous_repo = current_llm_repo

        # 1. Download FIRST. A wrong repo name or a network error then leaves the current model running.
        try:
            ensure_llm_downloaded(requested_repo)
        except Exception as de:
            model_switch_status["state"] = "error"
            model_switch_status["error"] = f"Download error for {requested_repo}: {de}. The current model is still loaded."
            logger.error(f"[switch-model] {model_switch_status['error']}")
            return

        logger.info(f"[switch-model] Unloading current pipeline ({previous_repo})...")
        llm_pipeline = None  # release reference so GPU memory can be reclaimed
        import gc
        gc.collect()
        time.sleep(1) # Give driver a moment to flush memory

        llm_dev, _ = check_inference_devices()
        load_llm_pipeline(device=llm_dev, repo_id=requested_repo)
        _update_active_model_metadata()

        # A GPU model that does not fit silently falls back to the (very slow) CPU. Treat that as a failure.
        fell_back = llm_pipeline is not None and active_llm_device != llm_dev and not SWITCH_ALLOW_CPU
        if fell_back:
            failure = (f"{requested_repo} did not fit on the Intel {llm_dev} (it only ran on {active_llm_device}, "
                       f"which is far too slow). Pick a smaller model, or set SWITCH_ALLOW_CPU=on to allow CPU.")
            llm_pipeline = None

        if llm_pipeline is not None:
            model_switch_status["state"] = "ready"
            logger.info(f"[switch-model] ✓ Switch complete → {requested_repo} on {active_llm_device}")
        else:
            failure = failure if fell_back else (llm_init_error or "Unknown error during model load.")
            logger.error(f"[switch-model] ✗ Failed to load {requested_repo}: {failure}")
            # 2. Roll back so the server is never left without an LLM
            restored = False
            if previous_repo and previous_repo != requested_repo:
                logger.warning(f"[switch-model] Restoring previous model {previous_repo}...")
                gc.collect()
                load_llm_pipeline(device=llm_dev, repo_id=previous_repo)
                _update_active_model_metadata()
                restored = llm_pipeline is not None
            model_switch_status["state"] = "error"
            model_switch_status["error"] = failure + (f" Restored previous model {previous_repo}." if restored else "")

    threading.Thread(target=_do_switch, daemon=True).start()
    return {
        "status": "loading",
        "model": requested_repo,
        "message": "Model switch started. Poll GET /api/switch-model/status for progress."
    }


@app.get("/api/switch-model/status")
async def switch_model_status():
    """Returns the current model switch progress and active model info."""
    return {
        "switch_status": model_switch_status,
        "current_model": current_llm_repo,
        "llm_loaded": llm_pipeline is not None,
        "llm_hardware": active_llm_device,
    }


@app.post("/api/switch-embed-model")
async def switch_embed_model(req: SwitchEmbedModelRequest):
    """Hot-swap the active embedding model without restarting the service.

    NOTE: After switching, any downstream vector store that was indexed with the
    previous model's dimensions will be incompatible.  Re-index before querying.
    """
    global embed_model, embed_switch_status

    requested_repo = req.model.strip()
    if not requested_repo:
        raise HTTPException(status_code=400, detail="'model' field is required.")

    if embed_switch_status["state"] == "loading":
        return JSONResponse(
            status_code=409,
            content={
                "status": "busy",
                "message": "An embedding model switch is already in progress.",
                "model": embed_switch_status["model"]
            }
        )

    embed_switch_status["state"] = "loading"
    embed_switch_status["model"] = requested_repo
    embed_switch_status["error"] = None

    def _do_embed_switch():
        global embed_model, embed_switch_status
        logger.info(f"[switch-embed-model] Unloading current embedding model ({current_embed_repo})...")
        embed_model = None  # release reference
        import gc
        gc.collect()
        time.sleep(1) # Give driver a moment to flush memory

        _, embed_dev = check_inference_devices()
        load_embed_model(device=embed_dev, repo_id=requested_repo)

        # Refresh metadata aliases
        embed_short = current_embed_repo.split("/")[-1]
        _embed_entry = {
            "name": embed_short,
            "model": embed_short,
            "details": {"family": "bert", "parameter_size": "base" if "base" in embed_short.lower() else "small", "format": "openvino"}
        }
        for alias in [embed_short, "nomic-embed-text", "bge-small-en-v1.5", "bge-base-en-v1.5", current_embed_repo]:
            active_models_metadata[alias] = _embed_entry

        if embed_model is not None:
            embed_switch_status["state"] = "ready"
            logger.info(f"[switch-embed-model] ✓ Switch complete → {requested_repo}")
        else:
            embed_switch_status["state"] = "error"
            embed_switch_status["error"] = embed_init_error or "Unknown error during embedding model load."
            logger.error(f"[switch-embed-model] ✗ Failed to load {requested_repo}: {embed_switch_status['error']}")

    threading.Thread(target=_do_embed_switch, daemon=True).start()
    return {
        "status": "loading",
        "model": requested_repo,
        "message": "Embedding model switch started. Poll GET /api/switch-embed-model/status for progress."
    }


@app.get("/api/switch-embed-model/status")
async def switch_embed_model_status():
    """Returns current embedding model switch progress."""
    return {
        "switch_status": embed_switch_status,
        "current_model": current_embed_repo,
        "embed_loaded": embed_model is not None,
        "embed_hardware": active_embed_device,
    }

@app.post("/api/embeddings")
async def ollama_embeddings(req: OllamaEmbedRequest):
    """Ollama-compatible single embedding generation."""
    query = req.prompt or (req.input if isinstance(req.input, str) else req.input[0] if req.input else "")
    if not query:
        raise HTTPException(status_code=400, detail="Missing prompt or input text.")
    
    vec = await asyncio.to_thread(compute_embedding, query)
    return {"embedding": vec}

@app.post("/api/embed")
async def ollama_embed(req: OllamaEmbedRequest):
    """Ollama batch embed endpoint."""
    inputs = req.input or ([req.prompt] if req.prompt else [])
    if isinstance(inputs, str):
        inputs = [inputs]
    
    results = await asyncio.to_thread(lambda: [compute_embedding(t) for t in inputs])
    return {"embeddings": results}

@app.post("/api/chat")
async def ollama_chat(req: OllamaChatRequest):
    """Ollama-compatible chat completion endpoint supporting streaming."""
    if not llm_pipeline:
        detail_msg = f"LLM pipeline is not ready: {llm_init_error}" if llm_init_error else "LLM pipeline is not ready."
        raise HTTPException(status_code=503, detail=detail_msg)

    msg_dicts = [{"role": m.role, "content": m.content} for m in req.messages]
    formatted_prompt = format_chat_prompt(msg_dicts)

    if req.stream:
        async def stream_generator():
            token_queue = queue.Queue()
            
            def streamer(chunk: str) -> bool:
                # Filter out null bytes and control-character artifacts that
                # openvino_genai emits when the model hits its KV-cache limit
                # or generates EOS/pad tokens. These decode as 0x00, \x01 etc.
                if chunk and not any(ord(c) < 9 for c in chunk):
                    token_queue.put(chunk)
                return False

            def run_inference():
                try:
                    import openvino_genai as ov_genai
                    gen_config = ov_genai.GenerationConfig()
                    gen_config.max_new_tokens = MAX_NEW_TOKENS
                    locked_generate(formatted_prompt, gen_config, streamer)
                except Exception as ex:
                    logger.error(f"Inference error during stream: {ex}")
                finally:
                    token_queue.put(None)

            threading.Thread(target=run_inference, daemon=True).start()

            while True:
                token = await asyncio.to_thread(token_queue.get)
                if token is None:
                    # Final done message
                    yield json.dumps({
                        "model": req.model,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "message": {"role": "assistant", "content": ""},
                        "done": True
                    }) + "\n"
                    break
                
                yield json.dumps({
                    "model": req.model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {"role": "assistant", "content": token},
                    "done": False
                }) + "\n"

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")
    else:
        import openvino_genai as ov_genai
        gen_config = ov_genai.GenerationConfig()
        gen_config.max_new_tokens = MAX_NEW_TOKENS
        output = await asyncio.to_thread(locked_generate, formatted_prompt, gen_config)
        return {
            "model": req.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": output},
            "done": True
        }

@app.post("/api/generate")
async def ollama_generate(req: OllamaGenerateRequest):
    """Ollama-compatible completion generation endpoint."""
    if not llm_pipeline:
        detail_msg = f"LLM pipeline is not ready: {llm_init_error}" if llm_init_error else "LLM pipeline is not ready."
        raise HTTPException(status_code=503, detail=detail_msg)

    msgs = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    msgs.append({"role": "user", "content": req.prompt})
    prompt = format_chat_prompt(msgs)

    if req.stream:
        async def stream_generator():
            token_queue = queue.Queue()
            def streamer(chunk: str) -> bool:
                # Filter out null bytes and control-character artifacts that
                # openvino_genai emits when the model hits its KV-cache limit
                # or generates EOS/pad tokens. These decode as 0x00, \x01 etc.
                if chunk and not any(ord(c) < 9 for c in chunk):
                    token_queue.put(chunk)
                return False

            def run_inference():
                try:
                    import openvino_genai as ov_genai
                    gen_config = ov_genai.GenerationConfig()
                    gen_config.max_new_tokens = MAX_NEW_TOKENS
                    llm_pipeline.generate(prompt, gen_config, streamer)
                except Exception as ex:
                    logger.error(f"Inference error during stream: {ex}")
                finally:
                    token_queue.put(None)

            threading.Thread(target=run_inference, daemon=True).start()

            while True:
                token = await asyncio.to_thread(token_queue.get)
                if token is None:
                    yield json.dumps({
                        "model": req.model,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "response": "",
                        "done": True
                    }) + "\n"
                    break
                
                yield json.dumps({
                    "model": req.model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "response": token,
                    "done": False
                }) + "\n"

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")
    else:
        import openvino_genai as ov_genai
        gen_config = ov_genai.GenerationConfig()
        gen_config.max_new_tokens = MAX_NEW_TOKENS
        output = await asyncio.to_thread(locked_generate, prompt, gen_config)
        return {
            "model": req.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": output,
            "done": True
        }

# ------------------------------------------------------------------------------
# OpenAI-Compatible Endpoints (/v1/*)
# ------------------------------------------------------------------------------
@app.get("/v1/models")
async def openai_models():
    data = [
        {"id": m, "object": "model", "owned_by": "openvino"}
        for m in active_models_metadata.keys()
    ]
    return {"object": "list", "data": data}

@app.post("/v1/embeddings")
async def openai_embeddings(req: OpenAIEmbeddingRequest):
    inputs = [req.input] if isinstance(req.input, str) else req.input
    data = []
    for idx, text in enumerate(inputs):
        vec = await asyncio.to_thread(compute_embedding, text)
        data.append({
            "object": "embedding",
            "index": idx,
            "embedding": vec
        })
    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0}
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", "11434"))
    logger.info(f"Starting OpenVINO NPU Server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port, log_level="info")
