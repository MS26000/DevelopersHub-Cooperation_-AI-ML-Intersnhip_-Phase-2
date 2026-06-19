# Task 4: Context-Aware Chatbot Using LangChain + RAG

**DevelopersHub Corporation — AI/ML Engineering Advanced Internship**

---

## Objective

Build a conversational chatbot that:

1. **Retrieves relevant information** from a vectorized knowledge base before generating answers (Retrieval-Augmented Generation — RAG).
2. **Maintains conversational memory** across multiple turns so follow-up questions are understood in context.
3. Supports a **dual knowledge base** — user-uploaded PDFs and Wikipedia articles as fallback.

The chatbot is deployable as a Streamlit web app and runs entirely on free, CPU-friendly models.

---

## Tech Stack

| Component | Tool / Model |
|-----------|-------------|
| LLM | `google/flan-t5-base` (248M params, free, local CPU) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-d) |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Framework | LangChain 1.x (LCEL — LangChain Expression Language) |
| Memory | Manual chat history list (window of last 5 turns) |
| Document Loading | PyPDFLoader + direct `wikipedia` library |
| Deployment | Streamlit |

---

## Methodology / Approach

### 1. Document Ingestion
- **PDF documents** are loaded from a local `data/pdfs/` directory using `PyPDFLoader`.
- **Wikipedia articles** are fetched directly using the `wikipedia` Python library for 5 AI/ML topics: Artificial Intelligence, Machine Learning, Natural Language Processing, Deep Learning, and Transformer models.
- Both sources are merged into a single corpus.

### 2. Text Splitting & Chunking
- Documents are split using `RecursiveCharacterTextSplitter` with:
  - Chunk size: **512 characters**
  - Chunk overlap: **64 characters**
  - Separators: paragraph breaks → newlines → sentences → words

### 3. Embeddings & Vector Store
- Chunks are embedded using `all-MiniLM-L6-v2` (384-dimensional dense vectors, normalized).
- Embeddings are indexed in a **FAISS vector store**, saved locally and reloaded on subsequent runs.
- Retrieval uses **Maximum Marginal Relevance (MMR)** — top 4 chunks fetched from a candidate pool of 20 — to maximise both relevance and diversity.

### 4. RAG Chain (LCEL)
- Built using LangChain 1.x's LCEL `|` operator — replacing the deprecated `ConversationalRetrievalChain`.
- At query time, the chain:
  1. Retrieves the top-4 most relevant chunks via MMR.
  2. Formats the last 5 turns of chat history into a readable string.
  3. Fills a `PromptTemplate` with `{context}`, `{history}`, and `{question}`.
  4. Passes the formatted prompt to `flan-t5-base` for generation.

### 5. Conversational Memory
- Implemented as a **manual Python list** of `{role, content}` dictionaries.
- Replaces `ConversationBufferWindowMemory` (removed in LangChain 1.x).
- Each turn appends both the user question and the bot answer to the history list.

### 6. Evaluation
- 5 factual questions were evaluated against the knowledge base.
- Metrics tracked: **retrieval hit rate** (whether the expected source type appeared in retrieved chunks) and **average response latency**.

---

## Key Results and Observations

| Metric | Value |
|--------|-------|
| Total chunks indexed | Varies by corpus size |
| Embedding dimension | 384 |
| Retrieval method | FAISS MMR (top-4 of 20 candidates) |
| Retrieval hit rate | ~80–100% on AI/ML topics |
| Average response time | ~2–5 seconds (CPU, flan-t5-base) |
| Conversational turns demo | 5 multi-turn exchanges |

### Observations

- **Wikipedia as fallback** works reliably for general AI/ML knowledge when no PDFs are available; the `DisambiguationError` is handled gracefully by picking the first suggested page.
- **MMR retrieval** prevents the chatbot from returning near-identical chunks and improves the diversity of context fed to the LLM.
- **flan-t5-base** is lightweight and runs on CPU but has a limited context window (~512 tokens); it handles factual Q&A well but may truncate long contexts.
- **Manual memory management** using a Python list is more transparent and LangChain-version-agnostic than built-in memory classes.
- The FAISS index is saved locally so embeddings are not recomputed on every run — significantly reducing startup time.

---

## Skills Demonstrated

- Retrieval-Augmented Generation (RAG) pipeline construction
- Document embedding and FAISS vector indexing
- LangChain 1.x LCEL chain building
- Conversational memory design
- LLM integration with open-source models (Hugging Face)
- Multi-turn chatbot evaluation

---

## How to Run

```bash
# Install dependencies
pip install langchain langchain-core langchain-community langchain-huggingface \
            langchain-text-splitters faiss-cpu sentence-transformers \
            transformers accelerate wikipedia pypdf tiktoken matplotlib numpy

# (Optional) Add PDF files to:
#   data/pdfs/

# Run the notebook
jupyter notebook task4_rag_chatbot.ipynb

# Deploy as a web app
streamlit run streamlit_app.py
```

---

## Repository Structure

```
task4_rag_chatbot.ipynb       ← Main notebook
data/
  pdfs/                       ← Place your PDF documents here
vectorstore/
  faiss_index/                ← FAISS index (auto-generated)
rag_config.json               ← Pipeline configuration
```
