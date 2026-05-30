import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from data_loader import load_all_files
from vector_store import VectorStore
from rag_chain import answer

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_base = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_base, "data")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Assistant",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0d1117; color: #e6edf3; }

  /* Hide sidebar toggle & footer */
  [data-testid="collapsedControl"] { display: none; }
  footer, #MainMenu { visibility: hidden; }

  /* User bubble */
  .user-msg {
    background: linear-gradient(135deg, #1f4068, #1b6ca8);
    color: #fff;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 18%;
    font-size: 0.95rem;
    line-height: 1.6;
  }

  /* Bot bubble */
  .bot-msg {
    background: #161b22;
    border: 1px solid #30363d;
    color: #e6edf3;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 18% 8px 0;
    font-size: 0.95rem;
    line-height: 1.7;
  }

  /* Source pill */
  .src-pill {
    background: #21262d;
    border-left: 3px solid #58a6ff;
    color: #8b949e;
    border-radius: 6px;
    padding: 6px 12px;
    margin: 4px 0;
    font-size: 0.78rem;
  }
  .src-pill b { color: #58a6ff; }

  /* Sample question chips */
  div[data-testid="stButton"] > button {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 20px;
    color: #79c0ff;
    font-size: 0.82rem;
    padding: 6px 14px;
    width: 100%;
    text-align: left;
    transition: border-color 0.2s;
  }
  div[data-testid="stButton"] > button:hover {
    border-color: #58a6ff;
    color: #ffffff;
  }

  /* Progress bar area */
  .stProgress > div > div { background-color: #238636; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "messages": [], "chat_history": [],
    "store": None, "store_ready": False, "client": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Auto-build on first load ──────────────────────────────────────────────────
if not st.session_state.store_ready and OPENAI_API_KEY:
    with st.spinner("Loading and indexing your car datasets… (first run only)"):
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            store = VectorStore(client)
            chunks = load_all_files(DATA_DIR)
            store.build(chunks)
            st.session_state.client = client
            st.session_state.store = store
            st.session_state.store_ready = True
        except Exception as e:
            st.error(f"Failed to build index: {e}")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please add it to your .env file.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🚗 Car Buying Assistant")
st.caption("Ask anything about new or used car prices, specs, and comparisons.")
st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🚗 {msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("📚 Sources", expanded=False):
                for src in msg["sources"]:
                    score_pct = int(src.get("score", 0) * 100)
                    preview = src["content"][:200].replace("\n", " ")
                    st.markdown(
                        f'<div class="src-pill"><b>{src["source"]}</b> · '
                        f'{src["page"]} · {score_pct}% match<br>'
                        f'<small>{preview}…</small></div>',
                        unsafe_allow_html=True,
                    )

# ── Sample questions (only when chat is empty) ────────────────────────────────
SAMPLE_QUESTIONS = [
    "What is the on-road price of Maruti Swift ZXI in Chennai?",
    "Compare Hyundai Creta petrol variants under ₹15 lakh",
    "List all CNG variants of Maruti Swift with on-road price",
    "Show me Hyundai i20 variants and prices",
    "Diesel used cars under ₹4 lakh in CarDekho",
    "Find used BMW cars from the global dataset",
]

if not st.session_state.messages:
    st.markdown("Try asking:")
    col1, col2 = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        col = col1 if i % 2 == 0 else col2
        if col.button(q, key=f"sq_{i}"):
            st.session_state._pending_q = q
            st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
pending = st.session_state.pop("_pending_q", None) if hasattr(st.session_state, "_pending_q") else None

if st.session_state.store_ready:
    user_input = pending or st.chat_input("Ask about car prices, specs, comparisons…")
else:
    st.chat_input("Indexing datasets, please wait…", disabled=True)
    user_input = None

# ── Clear chat button ─────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("Clear chat", key="clear"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ── Process input ─────────────────────────────────────────────────────────────
if user_input and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Searching…"):
        try:
            reply, sources = answer(
                query=user_input,
                store=st.session_state.store,
                client=st.session_state.client,
                chat_history=st.session_state.chat_history,
                top_k=6,
            )
            st.session_state.chat_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": reply},
            ])
            st.session_state.messages.append({
                "role": "assistant", "content": reply, "sources": sources,
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant", "content": f"Error: {e}", "sources": [],
            })
    st.rerun()