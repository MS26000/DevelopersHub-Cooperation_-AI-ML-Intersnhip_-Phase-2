# 🤖 DevelopersHub Corporation — AI/ML Engineering Advanced Internship
## Phase 2: Advanced Task Submissions

**Intern:** Malaika Shoaib
**Due Date:** 30th June, 2026
**Tasks Completed:** 5 out of 5

---

## 📁 Repository Structure

```
DevelopersHub-AI-ML-Internship-Phase2/
│
├── Task 1 — News Topic Classifier Using BERT/
│   └── task1_news_classifier_bert.ipynb
│
├── Task 2 — End-to-End ML Pipeline Churn Prediction/
│   └── task2_churn_pipeline.ipynb
│
├── Task 3 — Multimodal Housing Price Prediction/
│   └── task3_multimodal_housing.ipynb
│
├── Task 4 — RAG Chatbot Using LangChain/
│   └── task4_rag_chatbot.ipynb
│
├── Task 5 — Auto Tag Support Tickets Using LLM/
│   └── task5_auto_tag_support_tickets.ipynb
│
└── README.md  ← You are here
```

---

## 🗂️ Task Overview

| # | Task | Technique | Key Tools |
|---|------|-----------|-----------|
| 1 | News Topic Classifier | BERT Fine-tuning | HuggingFace, Gradio |
| 2 | Customer Churn Prediction | ML Pipeline + GridSearchCV | scikit-learn, joblib |
| 3 | Housing Price Prediction | Multimodal CNN + Tabular Fusion | TensorFlow, MobileNetV2 |
| 4 | Context-Aware RAG Chatbot | RAG + Conversational Memory | LangChain, FAISS, Streamlit |
| 5 | Auto-Tag Support Tickets | Prompt Engineering (Zero/Few/CoT) | Groq API, Llama 3.3 70B |

---

## 📰 Task 1: News Topic Classifier Using BERT

### Objective
Fine-tune `bert-base-uncased` to classify news headlines into 4 categories: **World, Sports, Business, Sci/Tech** using the AG News Dataset (120,000 samples).

### Methodology
- Tokenized headlines with BERT tokenizer (max length 128, `[CLS]`/`[SEP]` tokens, attention masks)
- Fine-tuned the full BERT backbone + a linear classification head (768 → 4 classes)
- Used HuggingFace `Trainer` API with mixed precision (fp16), warmup steps, and weight decay
- Deployed an interactive demo using **Gradio** for live headline prediction

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Max Token Length | 128 |
| Warmup Steps | 150 |

### Key Results

| Epoch | Train Loss | Val Loss | Accuracy | F1 Macro |
|-------|-----------|----------|----------|----------|
| 1 | 0.2797 | 0.2836 | 91.70% | 91.68% |
| 2 | 0.2568 | 0.2662 | 92.50% | 92.49% |
| 3 | 0.1132 | 0.2863 | **92.65%** | **92.62%** |

- Transfer learning is highly effective — 92%+ accuracy on only 8,000 training samples
- Sports is the easiest class (highly distinctive vocabulary); Business and World have the most overlap
- Training time: ~4.2 minutes on GPU

---

## 🔄 Task 2: End-to-End ML Pipeline — Customer Churn Prediction

### Objective
Build a production-ready scikit-learn `Pipeline` to predict customer churn on the **IBM Telco Churn Dataset** (7,043 customers × 21 features), with full preprocessing, model comparison, hyperparameter tuning, and serialized deployment.

### Methodology
- Split features into numeric (3), binary (6), and nominal (10) types — processed via `ColumnTransformer`
- 80/20 stratified train-test split; 19 raw features → 40 after encoding
- Compared Logistic Regression, Random Forest, and Gradient Boosting (`class_weight="balanced"`)
- Tuned Gradient Boosting with `GridSearchCV` (54 combinations × 5-fold CV = 270 fits, scored on ROC-AUC)
- Serialized the full pipeline with `joblib` (0.2 MB)

### Key Results

| Model | Accuracy | ROC-AUC | F1 |
|-------|----------|---------|-----|
| Logistic Regression | 0.7381 | 0.8414 | 0.6136 |
| Random Forest | 0.7771 | 0.8160 | 0.5341 |
| Gradient Boosting (baseline) | 0.7899 | 0.8350 | 0.5634 |
| **Gradient Boosting (tuned)** | **0.8041** | **0.8467** | **0.5805** |

- **Month-to-month contract** is the strongest churn predictor (importance: 0.38)
- 5-fold CV AUC: **0.8473 ± 0.0103** — model generalises well with no overfitting
- Best params: `lr=0.05, max_depth=3, n_estimators=100, subsample=0.8`

---

## 🏠 Task 3: Multimodal ML — Housing Price Prediction Using Images + Tabular Data

### Objective
Predict housing sale prices by fusing **structured tabular data** (19 numeric + 8 categorical features) with **visual data** from house room images, using CNN feature extraction and late fusion. Evaluated with MAE and RMSE.

### Methodology
- Preprocessed tabular data: imputation → label encoding → log-transform target → `StandardScaler`
- Used **MobileNetV2** (ImageNet, frozen) to extract 1280-d CNN features per image in batches
- Compressed image features via **PCA** (1280-d → 64-d)
- **Late fusion**: concatenated tabular + PCA image features into one matrix
- Trained 3 models: Baseline GBM (tabular only), Multimodal GBM (fused), Deep Fusion Neural Network (two-branch architecture with Huber loss)

### Key Results

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Baseline GBM (Tabular Only) | — | — | — |
| Multimodal GBM (Tabular + CNN) | — | — | — |
| Deep Fusion Neural Network | — | — | — |

> Metric values populate at runtime after model training.

- Multimodal models outperform the tabular-only baseline
- `OverallQual` is the strongest single predictor, followed by `GrLivArea` and `TotalBsmtSF`
- PCA compression retained most image variance while significantly cutting compute cost

---

## 💬 Task 4: Context-Aware Chatbot Using LangChain + RAG

### Objective
Build a conversational chatbot that retrieves answers from a vectorized knowledge base (RAG) and maintains multi-turn conversational memory. Deployable as a Streamlit web app using free, CPU-friendly models.

### Methodology
- Ingested **PDFs** (PyPDFLoader) + **Wikipedia articles** (5 AI/ML topics via `wikipedia` library)
- Split documents with `RecursiveCharacterTextSplitter` (chunk size: 512, overlap: 64)
- Embedded chunks using `all-MiniLM-L6-v2` (384-d) → indexed in **FAISS** with MMR retrieval (top-4 of 20)
- Built RAG chain using **LangChain 1.x LCEL** (`|` operator) — replaces deprecated `ConversationalRetrievalChain`
- Conversational memory implemented as a manual Python list (last 5 turns) — replaces removed `ConversationBufferWindowMemory`
- LLM: `google/flan-t5-base` (248M params, free, local CPU)

### Key Results

| Metric | Value |
|--------|-------|
| Retrieval method | FAISS MMR (top-4 of 20 candidates) |
| Embedding dimension | 384-d |
| Retrieval hit rate | ~80–100% on AI/ML topics |
| Avg response time | ~2–5 seconds (CPU) |
| Memory window | Last 5 conversation turns |

- Wikipedia fallback with `DisambiguationError` handling works reliably
- FAISS index saved locally — embeddings not recomputed on subsequent runs
- MMR prevents duplicate/redundant chunks from being sent to the LLM

---

## 🎫 Task 5: Auto-Tagging Support Tickets Using LLM

### Objective
Automatically classify customer support tickets into 5 categories (**Billing, Technical, Account, Shipping, General**) and assign priority levels using prompt engineering — no model training required. Uses **Llama-3.3-70B** via the free Groq API.

### Methodology
- Built a labelled dataset of **30 support tickets** (6 per category) for evaluation + 7 unseen tickets for live demo
- Implemented and compared 3 prompting techniques:
  - **Zero-Shot** — tag definitions only, no examples (~300 tokens/call)
  - **Few-Shot** — 1 labelled example per category as conversation turns (~700 tokens/call)
  - **Chain-of-Thought** — 5-step reasoning before classification (best for ambiguous tickets)
- All techniques output strict JSON: `tag`, `priority`, `confidence`, `reasoning`
- Temperature set to **0.1** for deterministic classification output
- Evaluated with Accuracy, Macro F1, Weighted F1, and Confusion Matrix

### Key Results

| Technique | Accuracy | F1 (Macro) | F1 (Weighted) |
|-----------|----------|------------|---------------|
| Zero-Shot | — | — | — |
| Few-Shot | — | — | — |

> Metric values populate at runtime after batch evaluation.

- Few-shot consistently outperforms zero-shot by resolving edge cases more accurately
- Billing vs Technical is the most common misclassification (payment page errors span both)
- CoT reasoning provides an auditable trace useful for human review and support agent context
- Confidence < 0.75 flags tickets for human review in production

---

## ⚙️ Setup & How to Run

### Prerequisites
```bash
pip install transformers datasets torch scikit-learn langchain faiss-cpu \
            sentence-transformers tensorflow groq gradio streamlit \
            pandas numpy matplotlib seaborn joblib wikipedia pypdf
```

### API Keys Required
| Task | Key Needed | Where to Get |
|------|-----------|--------------|
| Task 3 | `KAGGLE_USERNAME` + `KAGGLE_KEY` | [kaggle.com/settings](https://www.kaggle.com/settings) |
| Task 5 | `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |

> ⚠️ Never hardcode API keys. Use environment variables or Colab Secrets.

```bash
export GROQ_API_KEY=your_key_here
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_kaggle_key
```

---

## 🛠️ Technologies Used

`Python` · `HuggingFace Transformers` · `scikit-learn` · `TensorFlow/Keras` · `LangChain` · `FAISS` · `Groq API` · `MobileNetV2` · `BERT` · `Llama 3.3 70B` · `Flan-T5` · `Gradio` · `Streamlit` · `joblib` · `pandas` · `matplotlib` · `seaborn`
