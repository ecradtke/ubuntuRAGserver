import os
import re
import json
import subprocess
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range, PayloadSchemaType
import time
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None
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
LLAMA_URL_DEFAULT = os.getenv("LLAMA_URL", "http://127.0.0.1:8080")      # llama.cpp llama-server (chat)
EMBED_URL_DEFAULT = os.getenv("EMBED_URL", "http://127.0.0.1:11434")      # OpenVINO embedding server (NPU)
DEFAULT_QDRANT_HOST = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rss_articles")
MODELS_DIR = os.getenv("MODELS_DIR", "/var/lib/k13/models")               # GGUF files
LLM_SWITCH_BIN = os.getenv("LLM_SWITCH_BIN", "/usr/local/sbin/k13-llm-switch")
LLM_THINKING_ON = os.getenv("LLM_THINKING", "off").strip().lower() in ("on", "1", "true", "yes")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))

# ==============================================================================
# Sidebar Configuration
# ==============================================================================
st.sidebar.markdown("### ⚙️ Cluster Endpoints")

llama_url = st.sidebar.text_input(
    "Chat LLM Server (llama.cpp)",
    value=LLAMA_URL_DEFAULT,
    help="llama-server's OpenAI-compatible API on this server"
)
embed_url = st.sidebar.text_input(
    "Embedding Server (OpenVINO)",
    value=EMBED_URL_DEFAULT,
    help="OpenVINO embedding server (Intel NPU) on this server"
)
llama_api = llama_url.strip().rstrip("/")
embed_api = embed_url.strip().rstrip("/")

# Chat LLM telemetry
llm_connected = False
llm_model_name = ""
llm_ctx = None
try:
    _m = requests.get(f"{llama_api}/v1/models", timeout=3)
    if _m.status_code == 200:
        _data = _m.json().get("data", [])
        if _data:
            llm_model_name = _data[0].get("id", "")
            llm_connected = True
    _p = requests.get(f"{llama_api}/props", timeout=3)
    if _p.status_code == 200:
        llm_ctx = (_p.json().get("default_generation_settings") or {}).get("n_ctx")
except Exception:
    pass

# Embedding telemetry
embed_connected = False
embed_hw = "NPU"
_current_embed_model = "BAAI/bge-small-en-v1.5"
try:
    _h = requests.get(f"{embed_api}/healthz", timeout=3)
    if _h.status_code == 200:
        _hj = _h.json()
        embed_connected = bool(_hj.get("embed_loaded", False))
        embed_hw = _hj.get("embed_hardware", "NPU")
        _current_embed_model = _hj.get("embed_model", _current_embed_model)
except Exception:
    pass

ollama_connected = llm_connected          # header status uses this name
is_openvino = embed_connected
engine_hardware_info = f"llama.cpp ✓ • OpenVINO {embed_hw} {'✓' if embed_connected else '✗'}"
if embed_connected and embed_hw == "CPU":
    st.sidebar.warning("⚠️ Embeddings are running on the CPU (the NPU was not available).")

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
    max_value=50,
    value=6,
    step=1,
    help="Number of vector-matched article excerpts fed to the LLM as RAG context. Higher values give broader coverage but increase prompt size. Context is automatically trimmed to fit the model's window. Recommended: 5–15 for focused answers, 20–50 for broad research sweeps."
)

# ==============================================================================
# Sidebar: Live LLM Model Switcher (GGUF files for llama.cpp)
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Switch LLM Model")

_local_models = []
try:
    for _e in sorted(os.scandir(MODELS_DIR), key=lambda e: e.name.lower()):
        if _e.is_file() and _e.name.lower().endswith(".gguf"):
            _local_models.append((_e.name, _e.stat().st_size))
except Exception:
    pass
_current_server_model = llm_model_name
_sizes = dict(_local_models)
_names = [n for n, _ in _local_models]
_active_file = next((n for n in _names if n[:-5] == llm_model_name), None)

if llm_connected:
    st.sidebar.caption(f"🟢 Running: **{llm_model_name}**" + (f"  •  context {llm_ctx:,} tokens" if llm_ctx else ""))
else:
    st.sidebar.caption("🔴 llama-server is not responding (no model selected yet, or still loading).")

if _names:
    switch_target = st.sidebar.selectbox(
        "Target Model",
        _names,
        index=_names.index(_active_file) if _active_file in _names else 0,
        format_func=lambda n: f"{n[:-5]}  •  {_sizes[n] / 1024**3:.1f} GB",
        help=f"GGUF files found in {MODELS_DIR}. Switching restarts llama-server and takes up to a minute or two.",
        key="model_switch_target",
    )
    if switch_target == _active_file and llm_connected:
        st.sidebar.success("✅ Already active")
    elif st.sidebar.button("🔄 Switch Model", use_container_width=True, key="btn_switch_model"):
        with st.sidebar:
            with st.spinner("Restarting llama-server with the new model…"):
                _r = subprocess.run(["sudo", "-n", LLM_SWITCH_BIN, switch_target], capture_output=True, text=True, timeout=60)
                _ok = False
                if _r.returncode == 0:
                    for _ in range(90):
                        time.sleep(2)
                        try:
                            _chk = requests.get(f"{llama_api}/v1/models", timeout=3)
                            if _chk.status_code == 200 and any(m.get("id") == switch_target[:-5] for m in _chk.json().get("data", [])):
                                _ok = True
                                break
                        except Exception:
                            pass
        if _ok:
            st.rerun()
        elif _r.returncode != 0:
            st.sidebar.error(f"Switch failed: {(_r.stderr or _r.stdout).strip()[:200]}")
        else:
            st.sidebar.error("llama-server did not come up with that model. It may be too large for the GPU; check: ./scripts/k13ctl.sh logs llama")
else:
    st.sidebar.info(f"No .gguf models in {MODELS_DIR}. Download one on the server:\n\n"
                    "`./scripts/download-model.sh <hf-repo> <file.gguf> --switch`")

# Fetch live EMBED switch state from inference server
_current_embed_model = "BAAI/bge-small-en-v1.5"
_embed_switch_state = "idle"
_embed_switch_error = None
try:
    _em_resp = requests.get(f"{embed_api}/api/switch-embed-model/status", timeout=3)
    if _em_resp.status_code == 200:
        _em_json = _em_resp.json()
        _current_embed_model = _em_json.get("current_model", "BAAI/bge-small-en-v1.5")
        _embed_switch_state = _em_json.get("switch_status", {}).get("state", "idle")
        _embed_switch_error = _em_json.get("switch_status", {}).get("error")
except Exception:
    pass

_embed_current_label = _current_embed_model.split("/")[-1]
if _embed_switch_state == "loading":
    st.sidebar.caption("⏳ Switching embedding model…")
else:
    st.sidebar.caption(f"🟢 Embed Running: **{_embed_current_label}**")

# ==============================================================================
# Sidebar: Embedding Model Switcher
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Switch Embedding Model")

CURATED_EMBED_MODELS = {
    "BAAI/bge-small-en-v1.5": "BGE Small  •  384 dims  •  ~90 MB  (default, NPU)",
    "BAAI/bge-base-en-v1.5":  "BGE Base ⭐  •  768 dims  •  ~430 MB  (better quality)",
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
            _er = requests.post(f"{embed_api}/api/switch-embed-model", json={"model": embed_switch_target}, timeout=5)
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
            _er = requests.post(f"{embed_api}/api/switch-embed-model", json={"model": embed_switch_target}, timeout=5)
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
    <p class="header-subtitle">llama.cpp (chat LLM) • Intel OpenVINO NPU (embeddings) & Neural Vector Search grounded on local Ubuntu services</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    status_class = "status-online" if ollama_connected else "status-offline"
    card1_label = "Inference (llama.cpp + OpenVINO)"
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
    display_model_name = llm_model_name or "No model loaded"
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
                    clean_url = target_url.rstrip('/.,);')
                    try:
                        pts, _ = q_client.scroll(
                            collection_name=selected_collection,
                            scroll_filter=Filter(must=[FieldCondition(key="url", match=MatchValue(value=clean_url))]),
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
                                    "url": p.get("url", clean_url),
                                    "feed": p.get("feed", "Exact URL Match")
                                })
                    except Exception:
                        pass

            # Determine vector embedding input
            embed_input = search_keywords if search_keywords.strip() else active_query

            # Generate the query embedding on the OpenVINO NPU server
            _er = requests.post(f"{embed_api}/api/embeddings",
                                json={"model": _current_embed_model, "prompt": embed_input}, timeout=120)
            _er.raise_for_status()
            query_vector = _er.json()["embedding"]

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

        # Construct augmented user prompt.
        # Strategy: trim each excerpt AND clean each URL individually so the total
        # context stays within Phi-3.5-mini's 4K token window at any Top K value.
        # Google redirect URLs (google.com/url?...&url=<target>&...) are stripped to
        # just the inner target URL — they were contributing ~300 chars each.
        EXCERPT_MAX_CHARS = 300

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

        if retrieved_docs:
            context_parts = [
                f"Source [{meta.get('title', f'Doc {i+1}')} - {meta.get('feed', '')}]"
                f" (URL: {clean_url(meta.get('url', 'N/A'))}): {trim_excerpt(doc)}"
                for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_meta))
            ]
            context_str = "\n\n---\n\n".join(context_parts)
            augmented_user_prompt = f"Context:\n{context_str}\n\nQuestion: {active_query}"
        else:
            augmented_user_prompt = active_query

        # Stream the answer from llama.cpp (OpenAI-compatible SSE)
        message_placeholder = st.empty()
        full_response = ""
        raw_response = ""
        reasoning_seen = False

        try:
            _payload = {
                "model": llm_model_name or "local",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": augmented_user_prompt},
                ],
                "stream": True,
                "max_tokens": MAX_NEW_TOKENS,
                "temperature": 0.3,
                # Qwen3-style templates: answer directly instead of spending the budget on hidden reasoning
                "chat_template_kwargs": {"enable_thinking": LLM_THINKING_ON},
            }
            with requests.post(f"{llama_api}/v1/chat/completions", json=_payload, stream=True, timeout=(5, 900)) as _resp:
                _resp.raise_for_status()
                for _line in _resp.iter_lines(decode_unicode=True):
                    if not _line or not _line.startswith("data:"):
                        continue
                    _chunk = _line[5:].strip()
                    if _chunk == "[DONE]":
                        break
                    try:
                        _delta = json.loads(_chunk)["choices"][0].get("delta", {})
                    except Exception:
                        continue
                    if _delta.get("reasoning_content"):
                        reasoning_seen = True
                    _token = _delta.get("content") or ""
                    if not _token:
                        if reasoning_seen and not raw_response:
                            message_placeholder.markdown("💭 *Thinking…*")
                        continue
                    raw_response += _token
                    full_response, thinking = strip_thinking(raw_response, streaming=True)
                    if full_response:
                        message_placeholder.markdown(full_response + "▌")
                    else:
                        message_placeholder.markdown("💭 *Thinking…*")

            full_response, still_thinking = strip_thinking(raw_response)
            if not full_response:
                if still_thinking or reasoning_seen:
                    full_response = ("⚠️ The model spent its entire response on internal reasoning and never wrote an answer. "
                                     "Lower 'Excerpts to Retrieve (Top K)', ask a narrower question, or switch to a non-reasoning model.")
                else:
                    full_response = "⚠️ The model produced no answer. Try again or pick a different model."
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Generation error from llama.cpp: {e}"
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
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources_to_save
        })
