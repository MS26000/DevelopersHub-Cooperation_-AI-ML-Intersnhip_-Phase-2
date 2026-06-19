"""
Task 4: Context-Aware RAG Chatbot — Streamlit App
DevelopersHub Corporation — AI/ML Engineering Advanced Internship
LangChain 1.x compatible (LCEL-based, no deprecated imports)

Run with:
    streamlit run streamlit_app.py
"""

import os
import time
import json
import tempfile
import warnings
import wikipedia
from pathlib import Path

import streamlit as st

# LangChain 1.x compatible imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# HuggingFace — direct T5 inference (no pipeline task string needed)
from transformers import T5ForConditionalGeneration, T5Tokenizer

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL     = "google/flan-t5-base"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
RETRIEVER_K   = 4
MEMORY_K      = 5
VECTOR_DIR    = Path("vectorstore")
VECTOR_DIR.mkdir(exist_ok=True)

QA_PROMPT_TEMPLATE = """You are a helpful, accurate, and concise AI assistant.
Use the retrieved context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make up facts.

Conversation History:
{history}

Retrieved Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["history", "context", "question"]
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot | Muhammad Shahzad",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 2.2rem; margin-bottom: 0.3rem; }
    .main-header p  { opacity: 0.8; font-size: 1rem; }
    .chat-user {
        background: #e8f4fd;
        border-left: 4px solid #2196F3;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .chat-bot {
        background: #f0faf0;
        border-left: 4px solid #4CAF50;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .source-badge {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 0.2rem;
        border: 1px solid #90caf9;
    }
    .source-badge.wiki {
        background: #fff3e0;
        color: #e65100;
        border-color: #ffcc02;
    }
    .status-ready   { color: #4CAF50; font-weight: bold; }
    .status-loading { color: #FF9800; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Cached Model Loading ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


@st.cache_resource(show_spinner=False)
def load_llm_model():
    """Load flan-t5-base directly — compatible with all transformers versions."""
    tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL)
    model     = T5ForConditionalGeneration.from_pretrained(LLM_MODEL)
    return tokenizer, model


def generate_answer(prompt_text: str, tokenizer, model, max_length: int = 256) -> str:
    """Run direct T5 inference — no pipeline task string required."""
    inputs  = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.3,
        repetition_penalty=1.2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ── Helper Functions ──────────────────────────────────────────────────────────
def format_history(history: list, k: int = MEMORY_K) -> str:
    recent = history[-(k * 2):] if len(history) > k * 2 else history
    if not recent:
        return "No previous conversation."
    lines = []
    for turn in recent:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def format_docs(docs: list) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata.get('source_type','unknown')} | "
        f"{d.metadata.get('title', 'N/A')}]\n{d.page_content}"
        for d in docs
    )


def load_pdfs_from_uploads(uploaded_files) -> list:
    docs = []
    for uf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        pages  = loader.load()
        for page in pages:
            page.metadata["source_type"] = "pdf"
            page.metadata["filename"]    = uf.name
        docs.extend(pages)
        os.unlink(tmp_path)
    return docs


def load_wikipedia(topics: list) -> list:
    docs = []
    wikipedia.set_lang("en")
    for topic in topics:
        try:
            page    = wikipedia.page(topic, auto_suggest=False)
            content = page.content[:8000]
            docs.append(Document(
                page_content=content,
                metadata={
                    "source_type": "wikipedia",
                    "title":       page.title,
                    "topic":       topic,
                    "source":      page.url
                }
            ))
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                page    = wikipedia.page(e.options[0], auto_suggest=False)
                content = page.content[:8000]
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source_type": "wikipedia",
                        "title":       page.title,
                        "topic":       topic,
                        "source":      page.url
                    }
                ))
            except Exception:
                pass
        except Exception:
            pass
    return docs


def build_vectorstore(docs: list) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks     = splitter.split_documents(docs)
    embeddings = load_embeddings()
    return FAISS.from_documents(chunks, embeddings), len(chunks)


# ── Session State Init ────────────────────────────────────────────────────────
for key, default in {
    "chat_history":   [],
    "vectorstore":    None,
    "retriever":      None,
    "tokenizer":      None,
    "llm_model":      None,
    "n_chunks":       0,
    "kb_sources":     [],
    "total_latency":  0.0,
    "n_queries":      0,
    "kb_ready":       False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 Context-Aware RAG Chatbot Powered by: Muhammad Shahzad</h1>
    <p>LangChain 1.x + FAISS + flan-t5-base &nbsp;|&nbsp;
       PDF Upload + Wikipedia &nbsp;|&nbsp; Conversational Memory</p>
    <small>DevelopersHub Corporation — AI/ML Engineering Internship Task 4</small>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Knowledge Base Setup")

    # PDF Upload
    st.subheader("📄 Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Wikipedia Topics
    st.subheader("🌐 Wikipedia Topics")
    wiki_input = st.text_area(
        "One topic per line",
        value="Artificial Intelligence\nMachine Learning\nNatural Language Processing\nDeep Learning\nTransformer (machine learning model)",
        height=130
    )
    wiki_topics = [t.strip() for t in wiki_input.strip().splitlines() if t.strip()]

    st.divider()

    # Build Button
    if st.button("🔨 Build Knowledge Base", type="primary", use_container_width=True):
        if not uploaded_files and not wiki_topics:
            st.error("Please upload PDFs or enter Wikipedia topics.")
        else:
            all_docs = []
            sources  = []

            with st.status("Building knowledge base...", expanded=True) as status:

                # Load LLM
                st.write("⏳ Loading LLM (flan-t5-base)...")
                tokenizer, llm_model = load_llm_model()
                st.session_state.tokenizer  = tokenizer
                st.session_state.llm_model  = llm_model
                st.write("✅ LLM ready")

                # Load embeddings
                st.write("⏳ Loading embedding model...")
                load_embeddings()
                st.write("✅ Embeddings ready")

                # Load PDFs
                if uploaded_files:
                    st.write(f"📄 Loading {len(uploaded_files)} PDF(s)...")
                    pdf_docs = load_pdfs_from_uploads(uploaded_files)
                    all_docs.extend(pdf_docs)
                    sources.append(f"{len(pdf_docs)} PDF pages")
                    st.write(f"✅ {len(pdf_docs)} pages loaded")

                # Load Wikipedia
                if wiki_topics:
                    st.write(f"🌐 Fetching {len(wiki_topics)} Wikipedia articles...")
                    wiki_docs = load_wikipedia(wiki_topics)
                    all_docs.extend(wiki_docs)
                    sources.append(f"{len(wiki_docs)} Wikipedia articles")
                    st.write(f"✅ {len(wiki_docs)} articles loaded")

                if not all_docs:
                    st.error("No documents loaded. Check your internet connection.")
                    status.update(label="❌ Failed", state="error")
                    st.stop()

                # Build FAISS
                st.write("🔍 Building FAISS vector store...")
                vectorstore, n_chunks = build_vectorstore(all_docs)
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": RETRIEVER_K, "fetch_k": 20}
                )
                st.write(f"✅ {n_chunks} chunks indexed")

                status.update(label="✅ Knowledge base ready!", state="complete")

            st.session_state.vectorstore   = vectorstore
            st.session_state.retriever     = retriever
            st.session_state.n_chunks      = n_chunks
            st.session_state.kb_sources    = sources
            st.session_state.kb_ready      = True
            st.session_state.chat_history  = []
            st.session_state.n_queries     = 0
            st.session_state.total_latency = 0.0
            st.success("Ready to chat!")

    # Status Panel
    st.divider()
    st.subheader("📊 Session Stats")
    is_ready = st.session_state.kb_ready

    st.markdown(
        f"**Status:** <span class='{'status-ready' if is_ready else 'status-loading'}'>"
        f"{'● Ready' if is_ready else '○ Not built'}</span>",
        unsafe_allow_html=True
    )

    if is_ready:
        avg_lat = (
            st.session_state.total_latency / st.session_state.n_queries
            if st.session_state.n_queries > 0 else 0
        )
        c1, c2 = st.columns(2)
        c1.metric("Chunks",  st.session_state.n_chunks)
        c2.metric("Queries", st.session_state.n_queries)
        st.metric("Avg Latency", f"{avg_lat:.1f}s")
        st.markdown("**Sources:**")
        for s in st.session_state.kb_sources:
            st.markdown(f"  • {s}")

    # Clear Chat
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history  = []
        st.session_state.n_queries     = 0
        st.session_state.total_latency = 0.0
        st.rerun()

    # Model Info
    with st.expander("ℹ️ Model Info"):
        st.markdown(f"""
        | Component | Details |
        |---|---|
        | **LLM** | flan-t5-base |
        | **Embeddings** | all-MiniLM-L6-v2 |
        | **Vector DB** | FAISS (MMR) |
        | **Memory** | Window (k={MEMORY_K}) |
        | **Chunk size** | {CHUNK_SIZE} chars |
        | **LangChain** | 1.x LCEL |
        """)


# ── Main Chat Area ────────────────────────────────────────────────────────────
col_chat, col_info = st.columns([3, 1])

with col_chat:
    st.subheader("💬 Chat")

    if not st.session_state.chat_history:
        if not is_ready:
            st.info("👈 Build the knowledge base using the sidebar first.")
        else:
            st.info("Knowledge base is ready! Ask me anything.")

    # Display conversation
    for turn in st.session_state.chat_history:
        st.markdown(
            f"<div class='chat-user'><strong>🧑 You:</strong> {turn['question']}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='chat-bot'><strong>🤖 Bot:</strong> {turn['answer']}</div>",
            unsafe_allow_html=True
        )

        # Source badges
        if turn.get("sources"):
            badges = ""
            for src in turn["sources"]:
                if src["type"] == "wikipedia":
                    badges += f"<span class='source-badge wiki'>🌐 Wikipedia: {src['title']}</span>"
                else:
                    badges += f"<span class='source-badge'>📄 PDF: {src['title']}</span>"
            st.markdown(
                f"<div style='margin-left:1rem;margin-bottom:0.5rem;'>{badges}</div>",
                unsafe_allow_html=True
            )

        if turn.get("latency"):
            st.caption(f"⏱️ {turn['latency']:.1f}s")
        st.markdown("---")

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question...",
            placeholder="e.g. What is machine learning? How does attention work?",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Send ➤", use_container_width=True, type="primary")

    if submitted and user_input.strip():
        if not st.session_state.kb_ready:
            st.error("Please build the knowledge base first.")
        else:
            retriever  = st.session_state.retriever
            tokenizer  = st.session_state.tokenizer
            llm_model  = st.session_state.llm_model

            with st.spinner("Thinking..."):
                t0 = time.time()

                # Retrieve relevant chunks
                source_docs = retriever.invoke(user_input)

                # Build prompt manually
                context = format_docs(source_docs)
                history = format_history(
                    [{"role": t["role"], "content": c}
                     for turn in st.session_state.chat_history
                     for t, c in [
                         ({"role": "user"},  turn["question"]),
                         ({"role": "bot"},   turn["answer"])
                     ]]
                )

                prompt_text = QA_PROMPT.format(
                    history=history,
                    context=context,
                    question=user_input
                )

                # Generate answer
                answer  = generate_answer(prompt_text, tokenizer, llm_model)
                elapsed = time.time() - t0

            # Deduplicate sources
            seen, sources = set(), []
            for doc in source_docs:
                src_type = doc.metadata.get("source_type", "unknown")
                title    = doc.metadata.get(
                    "title", doc.metadata.get("filename", "Unknown")
                )
                key = f"{src_type}::{title}"
                if key not in seen:
                    sources.append({"type": src_type, "title": title})
                    seen.add(key)

            st.session_state.chat_history.append({
                "role":     "user",
                "question": user_input,
                "answer":   answer,
                "sources":  sources,
                "latency":  elapsed
            })
            st.session_state.n_queries     += 1
            st.session_state.total_latency += elapsed
            st.rerun()


with col_info:
    st.subheader("📚 How It Works")
    st.markdown("""
    **Step 1 — Ingest**
    PDFs loaded page-by-page. Wikipedia fetched via direct API.

    **Step 2 — Chunk**
    Text split into 512-char chunks with 64-char overlap.

    **Step 3 — Embed**
    Each chunk → 384-d vector via `all-MiniLM-L6-v2`.

    **Step 4 — Index**
    Vectors stored in FAISS for fast similarity search.

    **Step 5 — Retrieve**
    Your question embedded → top-4 chunks via MMR.

    **Step 6 — Generate**
    `flan-t5-base` reads chunks + question + history → answer.

    **Step 7 — Memory**
    Last 5 turns stored for follow-up understanding.
    """)

    if st.session_state.chat_history:
        st.subheader("🕐 Recent Latencies")
        for i, turn in enumerate(st.session_state.chat_history[-5:][::-1]):
            lat = turn.get("latency", 0)
            bar = "█" * min(int(lat * 2), 20)
            st.caption(f"Q{len(st.session_state.chat_history)-i}: {lat:.1f}s {bar}")
