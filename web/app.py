import os
import re
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range, PayloadSchemaType
import time
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None
import ollama
import requests

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

# ==============================================================================
# Hide model "thinking" (reasoning models such as Qwen3 / DeepSeek-R1)
# ==============================================================================
_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str, streaming: bool = False) -> tuple[str, bool]:
    """Remove reasoning blocks from model output.

    Returns (visible_text, still_thinking). While streaming, an unfinished
    <think> block (or a partially received tag) is hidden and still_thinking
    is True so the UI can show a placeholder instead of the raw reasoning.
    """
    # Some templates put <think> in the prompt, so only "</think>" is emitted
    if _THINK_CLOSE in text.lower() and _THINK_OPEN not in text.lower().split(_THINK_CLOSE)[0]:
        text = re.split(re.escape(_THINK_CLOSE), text, maxsplit=1, flags=re.IGNORECASE)[1]
    text = _THINK_BLOCK_RE.sub("", text)
    thinking = False
    lower = text.lower()
    idx = lower.rfind(_THINK_OPEN)
    if idx != -1:  # opened but not closed yet
        text, thinking = text[:idx], True
    elif streaming:  # hold back a partially received "<thi..." tag
        for n in range(len(_THINK_OPEN) - 1, 0, -1):
            if lower.endswith(_THINK_OPEN[:n]):
                text = text[:-n]
                break
    return text.lstrip(), thinking


# ==============================================================================
# Timeframe helpers (preset ranges + precise custom range)
# ==============================================================================
CUSTOM_TIMEFRAME = "Custom range…"
TIMEFRAME_CHOICES = ["All Time", "Past 24 Hours", "Past 7 Days", "Past 30 Days", CUSTOM_TIMEFRAME]
_PRESET_SECONDS = {"Past 24 Hours": 24 * 3600, "Past 7 Days": 7 * 24 * 3600, "Past 30 Days": 30 * 24 * 3600}
_COMMON_TIMEZONES = ["America/Chicago", "America/New_York", "America/Denver", "America/Los_Angeles", "UTC"]


def detect_timezone_name() -> str:
    """Browser time zone when Streamlit exposes it, else PORTAL_TZ / TZ, else UTC."""
    try:
        name = getattr(st.context, "timezone", None)
        if name:
            return str(name)
    except Exception:
        pass
    return os.getenv("PORTAL_TZ") or os.getenv("TZ") or "UTC"


def get_zone(name: str):
    try:
        return ZoneInfo(name) if ZoneInfo else timezone.utc
    except Exception:
        return timezone.utc


def resolve_timeframe(choice: str, custom_start_ts=None, custom_end_ts=None, now_ts=None):
    """Return (gte, lte) epoch seconds for the published_at filter; None means unbounded."""
    now_ts = int(now_ts if now_ts is not None else time.time())
    if choice == CUSTOM_TIMEFRAME:
        return custom_start_ts, custom_end_ts
    if choice in _PRESET_SECONDS:
        return now_ts - _PRESET_SECONDS[choice], None
    return None, None


# ==============================================================================
# Page Configuration & Rich Custom Styling
# ==============================================================================
st.set_page_config(
    page_title="K13 AI Hub | Vector RAG Interface",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich dark glassmorphism design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main background */
    .stApp {
        background: radial-gradient(circle at 20% 20%, #111827 0%, #030712 100%);
        color: #f3f4f6;
    }

    /* Custom Header Container */
    .header-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .header-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .header-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 14px 18px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f8fafc;
    }
    .status-dot {
        height: 9px;
        width: 9px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    .status-online { background-color: #10b981; box-shadow: 0 0 8px #10b981; }
    .status-offline { background-color: #ef4444; box-shadow: 0 0 8px #ef4444; }

    /* Excerpt Box */
    .excerpt-card {
        background: rgba(15, 23, 42, 0.5);
        border-left: 3px solid #6366f1;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 12px;
    }

    /* Quick Prompt Chips */
    div.stButton > button {
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(30, 41, 59, 0.5);
        color: #cbd5e1;
        transition: all 0.2s ease;
        padding: 6px 14px;
        font-size: 0.85rem;
    }
    div.stButton > button:hover {
        border-color: #6366f1;
        color: #ffffff;
        background: rgba(99, 102, 241, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Environment Configuration & Defaults
# ==============================================================================
# Native services on this server (scripts/run-web.sh exports these URLs)
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")   # OpenVINO server (LLM + embeddings)
DEFAULT_QDRANT_HOST = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rss_articles")
# Retrieval limits. The context sent to the LLM is capped to a token budget so a large Top K cannot exhaust
# GPU memory (the KV cache grows with the prompt). Raise RAG_CONTEXT_TOKEN_BUDGET only after testing.
TOP_K_MAX = int(os.getenv("TOP_K_MAX", "100"))
RAG_CONTEXT_TOKEN_BUDGET = int(os.getenv("RAG_CONTEXT_TOKEN_BUDGET", "8000"))
EXCERPT_MAX_CHARS = 300
CHARS_PER_TOKEN = 3.5      # conservative for English text with URLs
TOKENS_PER_EXCERPT = 145   # typical: source header + 300-char excerpt + separator

# ==============================================================================
# RAG context helpers (module level so they can be unit-tested)
# ==============================================================================
def clean_url(raw_url: str) -> str:
    """Extract the real URL from a Google redirect wrapper, or return as-is."""
    if not raw_url or raw_url == "N/A":
        return raw_url
    if "google.com/url" in raw_url:
        from urllib.parse import urlparse, parse_qs
        try:
            params = parse_qs(urlparse(raw_url).query)
            if "url" in params:
                return params["url"][0]
        except Exception:
            pass
    return raw_url


def trim_excerpt(text: str, limit: int = EXCERPT_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    trimmed = text[:limit]
    last_period = trimmed.rfind(". ")
    if last_period > limit * 0.6:
        return trimmed[:last_period + 1]
    return trimmed.rstrip() + "…"


def build_context(docs, metas, budget_tokens: int = RAG_CONTEXT_TOKEN_BUDGET):
    """Join the retrieved excerpts into one context string that fits the token budget.

    Order is similarity order, so if there are too many excerpts we first shorten each one
    (down to ~100 chars) and only then drop the least relevant ones from the end.
    Returns (context_text, excerpts_used, chars_per_excerpt, was_trimmed).
    """
    budget_chars = int(budget_tokens * CHARS_PER_TOKEN)
    sep = "\n\n---\n\n"
    heads = [
        f"Source [{m.get('title', f'Doc {i+1}')} - {m.get('feed', '')}]"
        f" (URL: {clean_url(m.get('url', 'N/A'))}): "
        for i, m in enumerate(metas)
    ]

    def render(count, limit):
        return sep.join(heads[i] + trim_excerpt(docs[i], limit) for i in range(count))

    n = len(docs)
    for limit in (EXCERPT_MAX_CHARS, 240, 190, 150, 120, 100):
        text = render(n, limit)
        if len(text) <= budget_chars:
            return text, n, limit, limit < EXCERPT_MAX_CHARS
    # still too long at the minimum excerpt size: keep the most relevant ones that fit
    count = n
    while count > 1 and len(render(count, 100)) > budget_chars:
        count -= 1
    return render(count, 100), count, 100, True


def context_stats(docs, used, limit, budget_tokens: int = RAG_CONTEXT_TOKEN_BUDGET):
    """How many excerpts were cut to fit: by the per-excerpt cap, by the token budget, or dropped."""
    used_docs = docs[:used]
    return {
        "retrieved": len(docs),
        "used": used,
        "dropped": len(docs) - used,
        "cut_to_cap": sum(1 for d in used_docs if len(d) > EXCERPT_MAX_CHARS),
        "cut_for_budget": sum(1 for d in used_docs if len(d) > limit) if limit < EXCERPT_MAX_CHARS else 0,
        "excerpt_chars": limit,
        "budget_tokens": budget_tokens,
    }


def render_query_stats(st_stats):
    """Per-question report: generation speed, token usage, retrieval and truncation."""
    if not st_stats:
        return
    with st.expander("📊 Query statistics", expanded=False):
        g = st_stats.get("gen", {})
        r = st_stats.get("retrieval", {})
        t = st_stats.get("timing", {})
        est = "" if g.get("exact") else " (est.)"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tokens/s", f"{g.get('tps', 0):.1f}{est}")
        c2.metric("Output tokens", f"{g.get('out_tokens', 0):,}")
        c3.metric("Prompt tokens", f"{g.get('in_tokens', 0):,}")
        c4.metric("Time to 1st token", f"{g.get('ttft_s', 0):.1f}s")
        st.markdown(
            f"- **Total time:** {t.get('total_s', 0):.1f}s "
            f"(embed {t.get('embed_s', 0):.2f}s · search {t.get('search_s', 0):.2f}s · generate {t.get('gen_s', 0):.1f}s)\n"
            f"- **Model:** `{st_stats.get('model', '?')}` on **{g.get('device') or '?'}**\n"
            f"- **Stop reason:** {g.get('stop', '?')}"
            + (" ⚠️ hit the max-new-tokens limit, answer may be cut off" if g.get("stop") == "length" else "")
            + f"\n- **Hidden reasoning stripped:** {g.get('thinking_chars', 0):,} chars"
        )
        if r and not r.get("retrieved"):
            st.markdown(f"**Retrieval:** no excerpts retrieved (Top K {r.get('top_k', '?')}).")
        elif r:
            st.markdown(
                "**Retrieval & truncation**\n\n"
                "| Item | Count |\n|---|---|\n"
                f"| Top K requested | {r.get('top_k', '?')} |\n"
                f"| Excerpts retrieved | {r.get('retrieved', 0)} |\n"
                f"| Excerpts sent to the model | {r.get('used', 0)} |\n"
                f"| Dropped (over token budget) | {r.get('dropped', 0)} |\n"
                f"| Cut at {EXCERPT_MAX_CHARS}-char excerpt cap | {r.get('cut_to_cap', 0)} |\n"
                f"| Shortened further for budget | {r.get('cut_for_budget', 0)} |\n"
                f"| Final chars per excerpt | {r.get('excerpt_chars', '?')} |\n"
                f"| Context budget | ~{r.get('budget_tokens', 0):,} tokens |"
            )

DEFAULT_LLM_MODEL = os.getenv("DEFAULT_MODEL", "OpenVINO/Phi-3.5-mini-instruct-int4-ov")

# ==============================================================================
# Sidebar Configuration
# ==============================================================================
st.sidebar.markdown("### ⚙️ Cluster Endpoints")

ollama_url = st.sidebar.text_input(
    "Inference Server URL",
    value=DEFAULT_OLLAMA_HOST,
    help="URL of the OpenVINO NPU / Ollama-compatible server on this server"
)

# Connect to Inference Engine & Query OpenVINO Telemetry
ollama_api = ollama_url.strip().rstrip('/')
ollama_client = ollama.Client(host=ollama_api)
ollama_connected = False
available_models = []
engine_hardware_info = "Intel OpenVINO NPU"
is_openvino = False
llm_hw = "GPU"
embed_hw = "NPU"

try:
    h_resp = requests.get(f"{ollama_api.rstrip('/')}/healthz", timeout=3)
    if h_resp.status_code == 200:
        is_openvino = True
        h_json = h_resp.json()
        llm_hw = h_json.get("llm_hardware", "GPU")
        embed_hw = h_json.get("embed_hardware", "NPU")
        engine_hardware_info = f"OpenVINO ({llm_hw} • {embed_hw})"
except Exception:
    pass

try:
    models_resp = ollama_client.list()
    models_list = models_resp.get("models", []) if isinstance(models_resp, dict) else getattr(models_resp, "models", [])
    available_models = [
        m.get("name", m.get("model", "")) if isinstance(m, dict) else getattr(m, "model", getattr(m, "name", "")) 
        for m in models_list
    ]
    available_models = [m for m in available_models if m]
    if available_models:
        ollama_connected = True
    else:
        available_models = ["OpenVINO-Phi-3.5-mini-instruct-int4-ov", "OpenVINO-Llama-3.2-3B-NPU", "llama3.2:latest"]
except Exception:
    available_models = ["OpenVINO-Phi-3.5-mini-instruct-int4-ov", "OpenVINO-Llama-3.2-3B-NPU", "llama3.2:latest"]

if is_openvino and llm_hw == "CPU":
    st.sidebar.warning("⚠️ LLM is currently running on CPU fallback.")
    if st.sidebar.button("⚡ Force Reload on GPU", use_container_width=True, key="btn_reload_gpu"):
        try:
            requests.post(f"{ollama_api.rstrip('/')}/api/reload", timeout=15)
            time.sleep(1)
            st.rerun()
        except Exception as _re:
            st.sidebar.error(f"Reload error: {_re}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗄️ Vector Database Engine")

db_engine = st.sidebar.radio(
    "Active Database",
    ["Qdrant (Local)", "ChromaDB (Local File)"],
    index=0
)

q_client = None
qdrant_connected = False
chroma_connected = False
doc_count = 0
selected_collection = DEFAULT_COLLECTION
chroma_col = None

if db_engine == "Qdrant (Local)":
    qdrant_url = st.sidebar.text_input(
        "Qdrant URL",
        value=DEFAULT_QDRANT_HOST,
        help="URL of the Qdrant REST API on this server"
    )

    available_collections = ["rss_articles", "audio_transcripts"]
    try:
        q_client = QdrantClient(url=qdrant_url, timeout=30.0)
        cols_resp = q_client.get_collections()
        fetched_cols = [c.name for c in cols_resp.collections]
        if fetched_cols:
            available_collections = fetched_cols
            qdrant_connected = True
    except Exception:
        pass

    # Ensure rss_articles is default index
    default_idx = available_collections.index("rss_articles") if "rss_articles" in available_collections else 0
    selected_collection = st.sidebar.selectbox(
        "Qdrant Collection",
        available_collections,
        index=default_idx,
        help="Select 'rss_articles' for the news items, or 'audio_transcripts' for Whisper transcripts."
    )

    if qdrant_connected and q_client is not None:
        try:
            col_info = q_client.get_collection(selected_collection)
            doc_count = col_info.points_count or 0
        except Exception:
            doc_count = 0

        # Ensure payload indexes exist for filtering (idempotent — safe to call every time)
        try:
            q_client.create_payload_index(
                collection_name=selected_collection,
                field_name="published_at",
                field_schema=PayloadSchemaType.INTEGER
            )
        except Exception:
            pass
        try:
            q_client.create_payload_index(
                collection_name=selected_collection,
                field_name="feed",
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass
        try:
            q_client.create_payload_index(
                collection_name=selected_collection,
                field_name="group",
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass

else:
    chroma_path = st.sidebar.text_input("Chroma DB Path", value="./chroma_news_db")
    selected_collection = st.sidebar.text_input("Collection Name", value="news")
    if chromadb is not None:
        try:
            chroma_client = chromadb.PersistentClient(path=chroma_path)
            chroma_col = chroma_client.get_or_create_collection(name=selected_collection)
            doc_count = chroma_col.count()
            chroma_connected = True
        except Exception as e:
            st.sidebar.error(f"ChromaDB Error: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Intelligence & Search")
top_k = st.sidebar.slider(
    "Excerpts to Retrieve (Top K)",
    min_value=1,
    max_value=TOP_K_MAX,
    value=min(6, TOP_K_MAX),
    step=1,
    help=("Number of vector-matched article excerpts fed to the LLM as RAG context. Higher values give broader coverage "
          "but a longer prompt (slower first word, more GPU memory). The context is capped at "
          f"{RAG_CONTEXT_TOKEN_BUDGET:,} tokens: beyond that, excerpts are shortened and then the least relevant are dropped. "
          "Recommended: 5-15 for focused answers, 20-50 for broad research sweeps.")
)
_est_tokens = top_k * TOKENS_PER_EXCERPT
if _est_tokens > RAG_CONTEXT_TOKEN_BUDGET:
    st.sidebar.caption(f"⚠️ ≈{_est_tokens:,} tokens needed; excerpts will be shortened to fit the {RAG_CONTEXT_TOKEN_BUDGET:,}-token budget.")
else:
    st.sidebar.caption(f"≈{_est_tokens:,} tokens of context at this setting (budget {RAG_CONTEXT_TOKEN_BUDGET:,}).")

# ==============================================================================
# Sidebar: Live LLM Model Switcher
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Switch LLM Model")

# Known pre-converted OpenVINO INT4 LLM models on Hugging Face Hub
CURATED_OV_MODELS = {
    "OpenVINO/Phi-3.5-mini-instruct-int4-ov":  "Phi-3.5 Mini  •  3.8B  •  ~2.2 GB  (default)",
    "OpenVINO/Phi-4-mini-instruct-int4-ov":    "Phi-4 Mini ⭐  •  3.8B  •  ~2.5 GB  (recommended)",
    "OpenVINO/Qwen3-4B-int4-ov":               "Qwen3 4B  •  4B  •  ~2.8 GB  (coding + reasoning)",
    "OpenVINO/Llama-3.2-3B-Instruct-int4-ov": "Llama 3.2 3B  •  3B  •  ~2.0 GB  (Meta reliable)",
    "OpenVINO/gemma-3-4B-it-int4-ov":          "Gemma 3 4B  •  4B  •  ~2.5 GB  (Google reasoning)",
    "OpenVINO/Qwen2.5-7B-Instruct-int4-ov":    "Qwen2.5 7B Instruct ⭐  •  7B  •  ~4.3 GB  (news summaries, small KV cache = long context)",
    "OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov": "Mistral 7B v0.3  •  7B  •  ~4.2 GB  (clean, concise writing, 32k ctx)",
    "OpenVINO/Llama-3.1-8B-Instruct-int4-ov":  "Llama 3.1 8B Instruct  •  8B  •  ~4.8 GB  (128k ctx)",
    "OpenVINO/Qwen3-8B-int4-ov":               "Qwen3 8B 🔥  •  8B  •  ~5.5 GB  (best quality, needs 6GB+)",
    # 12-14B class: needs a lot of GPU-accessible memory. If a model does not fit on the GPU the switch is
    # rolled back and the previous model stays loaded. Repo names are the usual OpenVINO Hub names; if one
    # fails to download the sidebar shows the error and the current model stays loaded.
    "OpenVINO/gemma-3-12b-it-int4-ov":         "Gemma 3 12B  •  12B  •  ~7.5 GB  (needs >8 GB GPU memory)",
    "OpenVINO/Qwen3-14B-int4-ov":              "Qwen3 14B  •  14B  •  ~8.5 GB  (needs >9 GB GPU memory)",
    "OpenVINO/Qwen2.5-14B-Instruct-int4-ov":   "Qwen2.5 14B Instruct  •  14B  •  ~8.5 GB  (needs >9 GB GPU memory)",
    "OpenVINO/phi-4-int4-ov":                  "Phi-4 14B  •  14B  •  ~8.5 GB  (needs >9 GB GPU memory, 16k ctx)",
}

# Pseudo-entry that reveals a text box for any other Hugging Face repo
CUSTOM_MODEL_KEY = "__custom__"
CURATED_OV_MODELS[CUSTOM_MODEL_KEY] = "✏️ Custom Hugging Face repo…"

# Fetch live switch state from the inference server
_current_server_model = DEFAULT_LLM_MODEL
_switch_state = "idle"
_switch_error = None
try:
    _sw_resp = requests.get(f"{ollama_api.rstrip('/')}/api/switch-model/status", timeout=3)
    if _sw_resp.status_code == 200:
        _sw_json = _sw_resp.json()
        _current_server_model = _sw_json.get("current_model", DEFAULT_LLM_MODEL)
        _switch_state = _sw_json.get("switch_status", {}).get("state", "idle")
        _switch_error = _sw_json.get("switch_status", {}).get("error")
except Exception:
    pass

_current_label = CURATED_OV_MODELS.get(_current_server_model, _current_server_model.split("/")[-1])
if _switch_state == "loading":
    st.sidebar.caption("⏳ Switching model…")
else:
    st.sidebar.caption(f"🟢 Running: **{_current_label}**")

_curated_keys = list(CURATED_OV_MODELS.keys())
_default_idx = _curated_keys.index(_current_server_model) if _current_server_model in _curated_keys else 0
switch_target = st.sidebar.selectbox(
    "Target Model",
    _curated_keys,
    index=_default_idx,
    format_func=lambda x: CURATED_OV_MODELS[x],
    help="Select an OpenVINO INT4 model to load. First switch downloads the model (~2–9 GB; make sure you have free disk and RAM). The server returns 503 during the switch — inference resumes automatically when ready.",
    key="model_switch_target"
)
if switch_target == CUSTOM_MODEL_KEY:
    switch_target = st.sidebar.text_input(
        "Hugging Face repo",
        placeholder="OpenVINO/<model>-int4-ov",
        help="Any OpenVINO-converted repo, e.g. one ending in -int4-ov.",
        key="model_switch_custom",
    ).strip()

# Fetch live EMBED switch state from inference server
_current_embed_model = "BAAI/bge-base-en-v1.5"
_embed_switch_state = "idle"
_embed_switch_error = None
try:
    _em_resp = requests.get(f"{ollama_api.rstrip('/')}/api/switch-embed-model/status", timeout=3)
    if _em_resp.status_code == 200:
        _em_json = _em_resp.json()
        _current_embed_model = _em_json.get("current_model", "BAAI/bge-base-en-v1.5")
        _embed_switch_state = _em_json.get("switch_status", {}).get("state", "idle")
        _embed_switch_error = _em_json.get("switch_status", {}).get("error")
except Exception:
    pass

_embed_current_label = _current_embed_model.split("/")[-1]
if _embed_switch_state == "loading":
    st.sidebar.caption("⏳ Switching embedding model…")
else:
    st.sidebar.caption(f"🟢 Embed Running: **{_embed_current_label}**")

if _switch_state == "loading":
    st.sidebar.info("⏳ Loading model — server will return 503 until complete.")
    if st.sidebar.button("↺ Check Status", use_container_width=True, key="btn_check_switch"):
        st.rerun()
elif _switch_state == "error":
    st.sidebar.error(f"❌ Switch failed: {_switch_error}")
    if st.sidebar.button("🔄 Retry Switch", use_container_width=True, key="btn_switch_model"):
        try:
            _r = requests.post(f"{ollama_api.rstrip('/')}/api/switch-model", json={"model": switch_target}, timeout=5)
            if _r.status_code == 200:
                time.sleep(0.5)
                st.rerun()
            else:
                st.sidebar.error(f"Server error {_r.status_code}: {_r.text[:120]}")
        except Exception as _e:
            st.sidebar.error(f"Connection error: {_e}")
elif not switch_target:
    st.sidebar.info("Enter a Hugging Face repo name above.")
elif switch_target == _current_server_model:
    st.sidebar.success("✅ Already active")
else:
    if st.sidebar.button("🔄 Switch Model", use_container_width=True, key="btn_switch_model"):
        try:
            _r = requests.post(f"{ollama_api.rstrip('/')}/api/switch-model", json={"model": switch_target}, timeout=5)
            if _r.status_code == 200:
                st.sidebar.success("🚀 Switch initiated!")
                time.sleep(0.5)
                st.rerun()
            elif _r.status_code == 409:
                st.sidebar.warning("⚠️ A switch is already in progress.")
            else:
                st.sidebar.error(f"Server error {_r.status_code}: {_r.text[:120]}")
        except Exception as _e:
            st.sidebar.error(f"Connection error: {_e}")

# ==============================================================================
# Sidebar: Embedding Model Switcher
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Switch Embedding Model")

CURATED_EMBED_MODELS = {
    "BAAI/bge-base-en-v1.5":  "BGE Base ⭐  •  768 dims  •  ~430 MB  (default, NPU)",
    "BAAI/bge-small-en-v1.5": "BGE Small  •  384 dims  •  ~90 MB  (faster, lower quality)",
}

_embed_keys = list(CURATED_EMBED_MODELS.keys())
_embed_default_idx = next((i for i, k in enumerate(_embed_keys) if k == _current_embed_model), 0)
embed_switch_target = st.sidebar.selectbox(
    "Target Embedding Model",
    _embed_keys,
    index=_embed_default_idx,
    format_func=lambda x: CURATED_EMBED_MODELS[x],
    help="BGE Base produces 768-dim vectors — switching requires re-indexing Qdrant! NPU compiles a static shape on first load.",
    key="embed_switch_target"
)

if _embed_switch_state == "loading":
    st.sidebar.info("⏳ Compiling embedding model — this may take 30–90 s.")
    if st.sidebar.button("↺ Check Embed Status", use_container_width=True, key="btn_check_embed"):
        st.rerun()
elif _embed_switch_state == "error":
    st.sidebar.error(f"❌ Embed switch failed: {_embed_switch_error}")
    if st.sidebar.button("🔄 Retry Embed Switch", use_container_width=True, key="btn_retry_embed"):
        try:
            _er = requests.post(f"{ollama_api.rstrip('/')}/api/switch-embed-model", json={"model": embed_switch_target}, timeout=5)
            if _er.status_code == 200:
                time.sleep(0.5)
                st.rerun()
        except Exception as _e:
            st.sidebar.error(f"Connection error: {_e}")
elif embed_switch_target == _current_embed_model:
    st.sidebar.success("✅ Already active")
else:
    _dims_change = ("bge-small" in _current_embed_model and "bge-base" in embed_switch_target) or \
                   ("bge-base" in _current_embed_model and "bge-small" in embed_switch_target)
    if _dims_change:
        st.sidebar.warning("⚠️ Dimension change detected — Qdrant collection must be re-indexed after switch.")
    if st.sidebar.button("🔬 Switch Embedding", use_container_width=True, key="btn_switch_embed"):
        try:
            _er = requests.post(f"{ollama_api.rstrip('/')}/api/switch-embed-model", json={"model": embed_switch_target}, timeout=5)
            if _er.status_code == 200:
                st.sidebar.success("🚀 Embedding switch initiated!")
                time.sleep(0.5)
                st.rerun()
            elif _er.status_code == 409:
                st.sidebar.warning("⚠️ A switch is already in progress.")
            else:
                st.sidebar.error(f"Server error {_er.status_code}: {_er.text[:120]}")
        except Exception as _e:
            st.sidebar.error(f"Connection error: {_e}")

st.sidebar.markdown("### 🔍 Search Filters")
timeframe_choice = st.sidebar.selectbox("Timeframe", TIMEFRAME_CHOICES)
timeframe_label = timeframe_choice
custom_start_ts = custom_end_ts = None

if timeframe_choice == CUSTOM_TIMEFRAME:
    _detected_tz = detect_timezone_name()
    _tz_options = list(dict.fromkeys([_detected_tz] + _COMMON_TIMEZONES))
    _tz_name = st.sidebar.selectbox("Time zone", _tz_options, help="Dates and times below are interpreted in this time zone.")
    _tz = get_zone(_tz_name)
    _now_local = datetime.now(_tz)

    st.session_state.setdefault("tf_start_date", (_now_local - timedelta(days=1)).date())
    st.session_state.setdefault("tf_start_time", (_now_local - timedelta(days=1)).time().replace(second=0, microsecond=0))
    st.session_state.setdefault("tf_end_date", _now_local.date())
    st.session_state.setdefault("tf_end_time", _now_local.time().replace(second=0, microsecond=0))

    st.sidebar.markdown("**From**")
    _c1, _c2 = st.sidebar.columns(2)
    _start_date = _c1.date_input("Start date", key="tf_start_date")
    _start_time = _c2.time_input("Start time", key="tf_start_time", step=60)
    st.sidebar.markdown("**To**")
    _c3, _c4 = st.sidebar.columns(2)
    _end_date = _c3.date_input("End date", key="tf_end_date")
    _end_time = _c4.time_input("End time", key="tf_end_time", step=60)

    _start_dt = datetime.combine(_start_date, _start_time, tzinfo=_tz)
    _end_dt = datetime.combine(_end_date, _end_time, tzinfo=_tz)
    if _start_dt > _end_dt:
        _start_dt, _end_dt = _end_dt, _start_dt
        st.sidebar.caption("↔️ Start was after end, so they were swapped.")
    custom_start_ts = int(_start_dt.timestamp())
    custom_end_ts = int(_end_dt.timestamp()) + 59  # include the whole end minute
    timeframe_label = f"{_start_dt:%b %d, %Y %H:%M} → {_end_dt:%b %d, %Y %H:%M} ({_tz_name})"
    st.sidebar.caption(f"🗓️ {timeframe_label}")

# Dynamically populate Feed dropdown from Qdrant if connected
feed_options = ["(All Feeds)"]
group_options = ["(All Groups)"]
if qdrant_connected and q_client is not None:
    try:
        # Use Qdrant facets to get distinct feed values
        feed_facet = q_client.facet(
            collection_name=selected_collection,
            key="feed",
            limit=200
        )
        for hit in feed_facet.hits:
            if hit.value and str(hit.value) not in feed_options:
                feed_options.append(str(hit.value))
        feed_options[1:] = sorted(feed_options[1:])
    except Exception:
        pass
    try:
        group_facet = q_client.facet(
            collection_name=selected_collection,
            key="group",
            limit=100
        )
        for hit in group_facet.hits:
            if hit.value and str(hit.value) not in group_options:
                group_options.append(str(hit.value))
        group_options[1:] = sorted(group_options[1:])
    except Exception:
        pass

feed_filter = st.sidebar.selectbox(
    "Feed",
    feed_options,
    help="Filter search results to a specific RSS feed source."
)
group_filter = st.sidebar.selectbox(
    "Group",
    group_options,
    help="Filter search results to a specific FreshRSS group/folder."
)
search_keywords = st.sidebar.text_input("Vector Search Keywords", value="", help="If provided, the database will search for these exact keywords instead of embedding your long chat prompt.")

persona_choice = st.sidebar.selectbox(
    "Analyst Persona",
    [
        "Objective News Analyst",
        "Executive Intelligence Brief",
        "Technical Deep Dive",
        "Analytical Unbiased"
    ]
)

if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ==============================================================================
# Header & Live Telemetry Metrics
# ==============================================================================
st.markdown("""
<div class="header-box">
    <h1 class="header-title">⚡ K13 AI Intelligence Hub</h1>
    <p class="header-subtitle">Intel OpenVINO Acceleration (GPU: LLM • NPU: Embeddings) & Neural Vector Search grounded on local Ubuntu services</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    status_class = "status-online" if ollama_connected else "status-offline"
    card1_label = "Inference Engine (Intel)" if is_openvino else "Inference Engine"
    status_text = engine_hardware_info if ollama_connected else "Unreachable"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{card1_label}</div>
        <div class="metric-value"><span class="status-dot {status_class}"></span>{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    is_online = qdrant_connected if db_engine.startswith("Qdrant") else chroma_connected
    status_class = "status-online" if is_online else "status-offline"
    status_text = f"{doc_count:,} Records" if is_online else "Unreachable"
    engine_name = f"Qdrant ({selected_collection})" if db_engine.startswith("Qdrant") else f"ChromaDB ({selected_collection})"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{engine_name}</div>
        <div class="metric-value"><span class="status-dot {status_class}"></span>{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    display_model_name = CURATED_OV_MODELS.get(_current_server_model, _current_server_model.split("/")[-1])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active Model</div>
        <div class="metric-value">{display_model_name}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Search Strategy</div>
        <div class="metric-value">Hybrid (Vector + URL)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Quick Suggestion Chips
st.markdown("##### 💡 Suggested Research Topics:")
chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)

preset_query = None
if chip_col1.button("🏛️ Minnesota Politics", use_container_width=True):
    preset_query = "Provide a briefing on recent Minnesota state politics, legislative developments, and government actions based on the reporting from the Star Tribune, Minnesota Reformer, and Session Daily."

if chip_col2.button("🛡️ Cybersecurity & CVEs", use_container_width=True):
    preset_query = "What are the latest high-impact cyber attacks, malware, and critical vulnerabilities reported by The Hacker News?"

if chip_col3.button("🚀 SpaceX & Tech", use_container_width=True):
    preset_query = "Summarize the recent tweets and posts from Elon Musk regarding SpaceX launches and tech updates."

if chip_col4.button("📈 Economy & Oil Markets", use_container_width=True):
    preset_query = "What are the latest reports and discussions regarding oil prices, shipping in the Strait of Hormuz, and economic indicators?"

# ==============================================================================
# Chat Interface & Persistent History
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("sources"):
            with st.expander(f"🔍 Retrieved {len(msg['sources'])} Context Excerpts", expanded=False):
                for s in msg["sources"]:
                    title = s.get("title", "Source Excerpt")
                    url = s.get("url", "")
                    date = s.get("date", "")
                    feed = s.get("feed", "")
                    header = f"**[{title}]({url})**" if url else f"**{title}**"
                    meta_tags = []
                    if feed:
                        meta_tags.append(feed)
                    if date:
                        meta_tags.append(date)
                    if meta_tags:
                        header += f" *({' • '.join(meta_tags)})*"
                    st.markdown(header)
                    st.caption(s.get("text", ""))
                    st.divider()
        st.markdown(msg["content"])
        render_query_stats(msg.get("stats"))

# Process input
user_input = st.chat_input("Ask about breaking events, summaries, or specific topics...")
active_query = preset_query if preset_query else user_input

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    retrieved_docs = []
    retrieved_meta = []

    # 1. Embed query & retrieve relevant news chunks
    with st.spinner(f"Executing search against {db_engine}..."):
        try:
            # Check for explicit URLs for exact matching
            detected_urls = re.findall(r'https?://[^\s"\'<>]+', active_query)
            exact_docs = []
            exact_meta = []

            # URL Lookup in Qdrant
            if db_engine.startswith("Qdrant") and qdrant_connected and q_client is not None and detected_urls:
                for target_url in detected_urls:
                    exact_url = target_url.rstrip('/.,);')
                    try:
                        pts, _ = q_client.scroll(
                            collection_name=selected_collection,
                            scroll_filter=Filter(must=[FieldCondition(key="url", match=MatchValue(value=exact_url))]),
                            limit=3,
                            with_payload=True
                        )
                        for pt in pts:
                            p = pt.payload or {}
                            t = p.get("text", p.get("text_snippet", p.get("summary", "")))
                            if t and t not in exact_docs:
                                exact_docs.append(t)
                                exact_meta.append({
                                    "title": p.get("title", "Exact Match"),
                                    "date": p.get("published_at", ""),
                                    "url": p.get("url", exact_url),
                                    "feed": p.get("feed", "Exact URL Match")
                                })
                    except Exception:
                        pass

            # Determine vector embedding input
            embed_input = search_keywords if search_keywords.strip() else active_query

            # Generate embedding via Ollama
            _t_embed0 = time.perf_counter()
            try:
                emb_resp = ollama_client.embed(model=_current_embed_model, input=embed_input)
                query_vector = emb_resp["embeddings"][0]
            except AttributeError:
                emb_resp = ollama_client.embeddings(model=_current_embed_model, prompt=embed_input)
                query_vector = emb_resp["embedding"]
            _t_embed1 = time.perf_counter()

            if db_engine.startswith("Qdrant") and qdrant_connected and q_client is not None:
                # Build Qdrant Filter
                q_must = []
                if feed_filter and feed_filter != "(All Feeds)":
                    q_must.append(FieldCondition(key="feed", match=MatchValue(value=feed_filter)))

                if group_filter and group_filter != "(All Groups)":
                    q_must.append(FieldCondition(key="group", match=MatchValue(value=group_filter)))

                tf_gte, tf_lte = resolve_timeframe(timeframe_choice, custom_start_ts, custom_end_ts)
                if tf_gte is not None or tf_lte is not None:
                    q_must.append(FieldCondition(key="published_at", range=Range(gte=tf_gte, lte=tf_lte)))

                q_filter = Filter(must=q_must) if q_must else None

                search_results = q_client.query_points(
                    collection_name=selected_collection,
                    query=query_vector,
                    query_filter=q_filter,
                    limit=top_k
                ).points

                # If filters returned nothing but a filter was active, retry without filter
                filters_were_active = q_filter is not None
                if not search_results and filters_were_active and timeframe_choice != CUSTOM_TIMEFRAME:
                    active_filter_desc = []
                    if feed_filter and feed_filter != "(All Feeds)": active_filter_desc.append(f"Feed: '{feed_filter}'")
                    if group_filter and group_filter != "(All Groups)": active_filter_desc.append(f"Group: '{group_filter}'")
                    if timeframe_choice != "All Time": active_filter_desc.append(f"Timeframe: {timeframe_label}")
                    st.warning(
                        f"⚠️ **No results matched your active filters** ({', '.join(active_filter_desc)}).  "
                        "Retrying without filters. The 'group' field will be populated on the next RSS sync run."
                    )
                    search_results = q_client.query_points(
                        collection_name=selected_collection,
                        query=query_vector,
                        limit=top_k
                    ).points

                if not search_results and timeframe_choice == CUSTOM_TIMEFRAME:
                    st.info(f"No articles were published in the selected range: {timeframe_label}.")

                for pt in search_results:
                    payload = pt.payload or {}
                    text = payload.get("text", payload.get("text_snippet", payload.get("summary", "")))
                    if text:
                        retrieved_docs.append(text)
                        raw_pub = payload.get("published_at", "")
                        if isinstance(raw_pub, (int, float)) and raw_pub > 0:
                            try:
                                from datetime import datetime, timezone
                                pub_date = datetime.fromtimestamp(raw_pub, tz=timezone.utc).strftime("%b %d, %Y")
                            except Exception:
                                pub_date = str(raw_pub)
                        else:
                            pub_date = str(raw_pub) if raw_pub else ""
                        retrieved_meta.append({
                            "title": payload.get("title", f"Record {pt.id}"),
                            "date": pub_date,
                            "url": payload.get("url", ""),
                            "feed": payload.get("feed", payload.get("filename", "Qdrant Index"))
                        })

            # Query ChromaDB
            elif db_engine.startswith("ChromaDB") and chroma_col is not None:
                search_results = chroma_col.query(
                    query_embeddings=[query_vector],
                    n_results=top_k
                )
                retrieved_docs = search_results["documents"][0] if search_results["documents"] else []
                retrieved_meta = search_results["metadatas"][0] if search_results["metadatas"] else []

            # Prepend exact URL matches
            if exact_docs:
                for edoc, emeta in zip(reversed(exact_docs), reversed(exact_meta)):
                    if edoc not in retrieved_docs:
                        retrieved_docs.insert(0, edoc)
                        retrieved_meta.insert(0, emeta)

        except Exception as e:
            st.error(f"Search error: {e}")

    # Display assistant response container
    with st.chat_message("assistant"):
        if retrieved_docs:
            with st.expander(f"🔍 Retrieved {len(retrieved_docs)} Context Excerpts", expanded=False):
                for idx, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_meta)):
                    title = meta.get("title", meta.get("source", f"Excerpt {idx+1}"))
                    timestamp = meta.get("date", meta.get("published", ""))
                    url = meta.get("url", "")
                    feed = meta.get("feed", "")
                    header = f"**[{title}]({url})**" if url else f"**{title}**"
                    meta_tags = []
                    if feed:
                        meta_tags.append(feed)
                    if timestamp:
                        meta_tags.append(timestamp)
                    if meta_tags:
                        header += f" *({' • '.join(meta_tags)})*"
                    st.markdown(header)
                    st.caption(doc)
                    st.divider()
        else:
            st.warning(
                f"⚠️ **No matching context retrieved** from {selected_collection} "
                f"(Total records indexed: {doc_count:,}). Generating answer without RAG context."
            )

        # Build System Persona Prompt
        if persona_choice == "Executive Intelligence Brief":
            system_prompt = (
                "You are an executive intelligence briefing officer. Provide an executive summary with key "
                "strategic takeaways, high-level bullets, and risk implications strictly using the provided context. "
                "Whenever you cite a source, include its URL as a Markdown link."
            )
        elif persona_choice == "Technical Deep Dive":
            system_prompt = (
                "You are a principal technical analyst. Deeply examine the architectural, technical, and factual "
                "details strictly using the provided context. Reference source titles and include their URLs as Markdown links where applicable."
            )
        elif persona_choice == "Analytical Unbiased":
            system_prompt = (
                "You are a rigorous, analytically-driven researcher with no political, ideological, or "
                "editorial bias. Your sole objective is to evaluate the provided source excerpts with "
                "intellectual honesty and methodical precision. Follow these principles strictly:\n"
                "1. BASE EVERY CLAIM on direct evidence from the retrieved context. Never introduce "
                "assumptions, prior beliefs, or information outside the provided excerpts.\n"
                "2. PRESENT MULTIPLE PERSPECTIVES where the sources contain differing or conflicting "
                "accounts. Do not favour one narrative over another without evidentiary justification.\n"
                "3. IDENTIFY GAPS AND CONFLICTS explicitly. If sources contradict each other or if "
                "insufficient data exists to answer the question, state this clearly and neutrally.\n"
                "4. SEPARATE FACTS FROM INTERPRETATION. Clearly distinguish between what the sources "
                "report as fact and what is editorial opinion, speculation, or projection.\n"
                "5. USE NEUTRAL, PRECISE LANGUAGE. Avoid emotive, sensational, or loaded phrasing. "
                "Quantify claims where possible (dates, figures, percentages).\n"
                "6. CITE YOUR SOURCES by referencing the source title and feed for every key claim, and include the URL as a Markdown link.\n"
                "If the retrieved context does not contain sufficient information to answer the question, "
                "state this directly rather than speculating."
            )
        else:
            system_prompt = (
                "You are an objective news research analyst. Answer the query using ONLY the "
                "provided source context. If the excerpts do not contain the answer, explicitly state "
                "that the database lacks sufficient reporting on the topic. Reference source titles and include their URLs as Markdown links where applicable."
            )

        # Construct the augmented user prompt (context is capped to RAG_CONTEXT_TOKEN_BUDGET, see build_context).
        if retrieved_docs:
            context_str, _used_k, _used_lim, _trimmed = build_context(retrieved_docs, retrieved_meta)
            _ctx_stats = context_stats(retrieved_docs, _used_k, _used_lim)
            if _trimmed:
                st.caption(f"ℹ️ Context trimmed to fit the ~{RAG_CONTEXT_TOKEN_BUDGET:,}-token budget: "
                           f"{_used_k} of {len(retrieved_docs)} excerpts, up to {_used_lim} characters each.")
            augmented_user_prompt = f"Context:\n{context_str}\n\nQuestion: {active_query}"
        else:
            augmented_user_prompt = active_query

        _t_search_end = time.perf_counter()
        _ctx_stats = locals().get('_ctx_stats') or {}
        _final_chunk = {}
        _t_first = None
        # Stream response from Ollama (without importing pandas)
        message_placeholder = st.empty()
        full_response = ""
        raw_response = ""

        try:
            stream = ollama_client.chat(
                model=_current_server_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": augmented_user_prompt}
                ],
                stream=True
            )

            _t_gen0 = time.perf_counter()
            for chunk in stream:
                if _t_first is None and chunk["message"]["content"]:
                    _t_first = time.perf_counter()
                if chunk.get("done"):
                    _final_chunk = dict(chunk) if not hasattr(chunk, "model_dump") else chunk.model_dump()
                raw_response += chunk["message"]["content"]
                full_response, thinking = strip_thinking(raw_response, streaming=True)
                if full_response:
                    message_placeholder.markdown(full_response + "▌")
                elif thinking or raw_response:
                    message_placeholder.markdown("💭 *Thinking…*")

            full_response, still_thinking = strip_thinking(raw_response)
            if not full_response:
                if still_thinking:
                    full_response = ("⚠️ The model spent its entire response on internal reasoning and never wrote an answer. "
                                     "Lower 'Excerpts to Retrieve (Top K)', ask a narrower question, or pick a non-reasoning model "
                                     "(Phi-4 Mini, Llama 3.2 3B).")
                else:
                    full_response = "⚠️ The model produced no answer. Try again or pick a different model."
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Generation error from OpenVINO: {e}"
            message_placeholder.error(full_response)

        # Save to session history
        sources_to_save = [
            {
                "title": meta.get("title", f"Excerpt {i+1}"),
                "url": meta.get("url", ""),
                "date": meta.get("date", ""),
                "feed": meta.get("feed", ""),
                "text": doc
            }
            for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_meta))
        ]
        _t_end = time.perf_counter()
        _gen_t0 = locals().get('_t_gen0', _t_search_end)
        _gen_s = max(_t_end - _gen_t0, 1e-9)
        _out = _final_chunk.get("eval_count")
        _in = _final_chunk.get("prompt_eval_count")
        _exact = bool(_out)
        if not _out:
            _out = int(len(raw_response) / CHARS_PER_TOKEN)
        if not _in:
            _in = int((len(system_prompt) + len(augmented_user_prompt)) / CHARS_PER_TOKEN)
        _ttft = (_final_chunk.get("prompt_eval_duration") or 0) / 1e9 or ((_t_first - _gen_t0) if _t_first else _gen_s)
        _decode_s = (_final_chunk.get("eval_duration") or 0) / 1e9 or max(_gen_s - _ttft, 1e-9)
        _thinking_chars = max(len(raw_response) - len(full_response), 0)
        query_stats = {
            "model": _current_server_model,
            "gen": {
                "tps": _out / _decode_s,
                "out_tokens": _out, "in_tokens": _in, "ttft_s": _ttft,
                "exact": _exact, "device": _final_chunk.get("device") or llm_hw,
                "stop": _final_chunk.get("done_reason", "?"),
                "thinking_chars": _thinking_chars,
            },
            "retrieval": dict(_ctx_stats, top_k=top_k),
            "timing": {
                "embed_s": _t_embed1 - _t_embed0 if '_t_embed1' in locals() else 0,
                "search_s": _t_search_end - _t_embed1 if '_t_embed1' in locals() else 0,
                "gen_s": _gen_s, "total_s": _t_end - _t_embed0 if '_t_embed0' in locals() else _gen_s,
            },
        }
        render_query_stats(query_stats)
        try:
            _log = os.getenv("QUERY_STATS_LOG", "").strip()
            if _log:
                import json as _json
                os.makedirs(os.path.dirname(_log) or ".", exist_ok=True)
                with open(_log, "a", encoding="utf-8") as _f:
                    _f.write(_json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "query": active_query, **query_stats}) + "\n")
        except Exception:
            pass
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources_to_save,
            "stats": query_stats
        })
