import os
import tempfile
import time
from typing import List, Dict, Any

import streamlit as st
from loguru import logger

# Optional audio recorder widget. If not available, file upload falls back.
try:
    from streamlit_audiorecorder import audiorecorder
    AUDIO_RECORDER_AVAILABLE = True
except Exception:
    AUDIO_RECORDER_AVAILABLE = False

# Transcription (STT)
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# TTS
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# LangChain and vectorstore
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None

try:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
except Exception:
    HuggingFaceEmbeddings = None
    FAISS = None
    Document = None

# For fallback simple LLM reply
import random
from dotenv import load_dotenv
load_dotenv()

# Dummy customer data
DUMMY_CUSTOMER = {
    "name": "Jane Doe",
    "account_balance": 15320.45,
    "last_transaction": "2025-10-18: Grocery - ₹1240",
    "due_bill_amount": 450.0,
    "next_due_date": "2025-12-01",
    "account_number": "XXXX-XXXX-5678",
}

# Logging
logger.remove()
logger.add(lambda msg: st.session_state.setdefault("_logs", []).append(msg), level="DEBUG")
logger.add(lambda msg: print(msg), level="INFO")

# Session State

def ensure_session():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("vectorstore", None)
    st.session_state.setdefault("llm_backend", "groq")
    st.session_state.setdefault("_logs", [])
    st.session_state.setdefault("last_audio", None)
    st.session_state.setdefault("sel", None)


# UI logs

def log_ui():
    with st.expander("Logs"):
        for l in st.session_state.get("_logs", [])[-200:]:
            st.text(l)


# Cache Whisper model
@st.cache_resource

def load_whisper_model():
    if WHISPER_AVAILABLE:
        logger.info("Loading Whisper tiny model...")
        return WhisperModel("tiny", device="cpu", compute_type='int8', cpu_threads=4)
    return None


# RAG Setup
@st.cache_resource

def init_vectorstore():
    if HuggingFaceEmbeddings is None or FAISS is None:
        logger.warning("RAG disabled: dependencies missing.")
        return None
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [
        Document(page_content="Reset PIN: Visit nearest ATM or mobile banking", metadata={"source": "faq"}),
        Document(page_content="You can transfer between accounts using UPI or NEFT", metadata={"source": "faq"}),
        Document(page_content="General banking queries handled by FAQ agent", metadata={"source": "faq"}),
        Document(
            page_content=(
                "To transfer money between accounts, say 'transfer' followed by amount and destination. "
                "For example, 'transfer five hundred to savings'. Transfers within the same bank are instant; "
                "external transfers may take up to 2 business days."
            ),
            metadata={"source": "transfer", "title": "Money transfer between accounts"},
        ),
        Document(
            page_content=(
                "If you forgot your PIN, say 'reset PIN'. We will send a one-time verification code to your "
                "registered mobile. After verifying, you can set a new MPIN. Do not share verification codes "
                "with anyone."
            ),
            metadata={"source": "pin", "title": "Reset PIN and authentication"},
        )
    ]
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    vs = FAISS.from_texts(texts, embed, metadatas=metas)
    logger.info("Vectorstore initialized with %d docs" % len(texts))
    return vs


def rag_query(query: str, k: int = 2):
    vs = st.session_state.vectorstore
    if vs is None:
        return []
    results = vs.similarity_search_with_score(query, k=k)
    return [{"content": d.page_content, "metadata": d.metadata, "score": s} for d, s in results]


# LLM Clients

def get_groq():
    if ChatGroq and os.getenv("GROQ_API_KEY"):
        logger.info("Using Groq")
        return ChatGroq(model="qwen/qwen3-32b", max_tokens=1000, temperature=0.2)
    return None


def get_ollama():
    try:
        if ChatOllama:
            return ChatOllama(model="llama3.2")
    except: pass
    return None


def llm_reply(prompt: str) -> str:
    if st.session_state.llm_backend == "groq":
        groq = get_groq()
        if groq:
            try:
                resp = groq.invoke(prompt)
                return resp.content
            except:
                st.session_state.llm_backend = "ollama"
    if st.session_state.llm_backend == "ollama":
        oll = get_ollama()
        if oll:
            try:
                resp = oll.invoke(prompt)
                return resp.content
            except:
                st.session_state.llm_backend = "fallback"
    return "Unable to access models — check Groq or Ollama setup."


# STT

# def transcribe_bytes(b: bytes):
#     model = load_whisper_model()
#     if model:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#             tmp.write(b)
#             tp = tmp.name
#     try:
#         segments, _ = model.transcribe(tp)
#         return " ".join([seg.text for seg in segments])
#     except:
#         pass
#     return "(Could not transcribe in this environment)"

def transcribe_bytes(b: bytes):
    """Robust transcription helper that writes bytes to a temp file and handles different whisper return shapes."""
    if not b:
        return "(No audio provided)"


    model = load_whisper_model()
    if model is None:
        logger.warning("Whisper model not available.")
        return "(Could not transcribe in this environment)"


    tp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b)
            tp = tmp.name

        # The faster-whisper wrapper returns either (segments, info) or a dictionary depending on call
        res = model.transcribe(tp)

        # If it's a dict with 'text'
        if isinstance(res, dict):
            if res.get("text"):
                return res["text"].strip()
            segments = res.get("segments") or []
            return " ".join(getattr(seg, "text", seg.get("text", "")) if isinstance(seg, dict) else getattr(seg, "text", "") for seg in segments).strip()

        # If it's (segments, info)
        if isinstance(res, (list, tuple)) and len(res) >= 1:
            segments = res[0]
            return " ".join(getattr(seg, "text", seg.get("text", "")) if isinstance(seg, dict) else getattr(seg, "text", "") for seg in segments).strip()

        # Fallback
        return str(res)

    except Exception as e:
        logger.exception("Transcription failed: %s", e)
        return "(Could not transcribe in this environment)"
    finally:
        try:
            if tp and os.path.exists(tp):
                os.remove(tp)
        except Exception:
            pass

# TTS

def synthesize_tts(text: str):
    try:
        from gtts import gTTS
        out = os.path.join(tempfile.gettempdir(), f"tts_{int(time.time()*1000)}.mp3")
        tts = gTTS(text=text, lang='en')
        tts.save(out)
        return out
    except:
        return None


# React Agent Routing

def route_agent(query: str) -> Dict[str, Any]:
    q = query.lower()


    faq_kw = ["reset", "how", "transfer", "generic", "faq"]
    pay_kw = ["bill", "due", "payment", "pay"]
    acc_kw = ["balance", "account", "bank", "statement"]


    if any(k in q for k in faq_kw):
        logger.info("Route → FAQ / RAG")
        ctx = rag_query(query)
        context = "".join([c["content"] for c in ctx]
                          )
        ans = llm_reply(f"Use FAQs:\n{context}\nUser: {query}\nAnswer:")
        return {"route": "faq", "answer": ans}


    if any(k in q for k in pay_kw):
        logger.info("Route → Payments")
        info = DUMMY_CUSTOMER
        ans = llm_reply(f"User asked about payments. Info: {info}. Q: {query}. Reply helpful.")
        return {"route": "payments", "answer": ans}


    if any(k in q for k in acc_kw):
        logger.info("Route → Accounts")
        info = DUMMY_CUSTOMER
        ans = llm_reply(f"User asked about account. Info: {info}. Q: {query}. Reply helpful.")
        return {"route": "accounts", "answer": ans}


    logger.info("Route → default fallback to FAQ")
    ans = llm_reply(query)
    return {"route": "generic", "answer": ans}

# Streamlit UI
st.set_page_config(page_title="AgnoIVR", layout="wide")
ensure_session()
st.title("AgnoIVR — Agentic Bank IVR Prototype")

col1, col2 = st.columns([2,3])

with col1:
    st.subheader("🎧 Record or Upload Audio → STT")
    if AUDIO_RECORDER_AVAILABLE:
        logger.info("Using audio recorder widget.")
        # audio_bytes = audiorecorder("Record audio", key="rec1")
        # if audio_bytes is not None and st.button("Submit recording"):
        #     text = transcribe_bytes(audio_bytes)
        #     st.session_state.messages.append({"role":"user","text":text})
        #     res = route_agent(text)
        #     st.session_state.messages.append({"role":"assistant","text":res["answer"]})
        audio_bytes = audiorecorder("Record audio", key="rec1")
        if audio_bytes:
        # Persist across reruns so Submit can access it
            st.session_state.last_audio = audio_bytes
            st.success("Recording captured — press Submit to transcribe.")
            st.audio(audio_bytes)


        if st.button("Submit recording"):
            if not st.session_state.last_audio:
                st.warning("No recording found. Record first.")
            else:
                with st.spinner("Transcribing..."):
                    text = transcribe_bytes(st.session_state.last_audio)
                st.session_state.messages.append({"role":"user","text":text})
                res = route_agent(text)
                st.session_state.messages.append({"role":"assistant","text":res["answer"]})
                # optional: clear last_audio if you want to force new recordings
                st.session_state.last_audio = None
                st.experimental_rerun()
    else:
        up = st.file_uploader("Upload Audio", type=["wav","mp3","m4a"])
        if up and st.button("Transcribe"):
            text = transcribe_bytes(up.read())
            st.session_state.messages.append({"role":"user","text":text})
            res = route_agent(text)
            st.session_state.messages.append({"role":"assistant","text":res["answer"]})

    st.subheader("⌨️ Type message")
    txt = st.text_input("Message")
    if st.button("Send") and txt.strip():
        st.session_state.messages.append({"role":"user","text":txt})
        res = route_agent(txt)
        st.session_state.messages.append({"role":"assistant","text":res["answer"]})

    st.subheader("🗣️ TTS (selected)")
    if st.button("Play Selected"):
        idx = st.session_state.get("sel")
        if idx is not None:
            p = synthesize_tts(st.session_state.messages[idx]["text"])
            if p: st.audio(p)

with col2:
    st.subheader("💬 Conversation")
    for i,m in enumerate(st.session_state.messages):
        r = m["role"].upper()
        st.markdown(f"**{r}**: {m['text']}")
        if st.button("Select", key=f"sel{i}"):
            st.session_state["sel"] = i
        st.divider()

log_ui()

if st.session_state.vectorstore is None:
    st.session_state.vectorstore = init_vectorstore()

st.caption("No OpenAI used. Powered by Groq, Ollama, Whisper, FAISS, Streamlit.")
