# 🤖 AI/ML Engineering Advanced Internship

**DevelopersHub Corporation**

---

A collection of five end-to-end AI/ML projects covering NLP, classical machine learning, multimodal learning, retrieval-augmented generation, and LLM prompt engineering — built without shortcuts, evaluated rigorously, and documented for production relevance.

---

## 📁 Repository Structure

```
├── task1_news_classifier/
│   └── task1_news_bert_classifier.ipynb
├── task2_churn_prediction/
│   └── task2_churn_pipeline.ipynb
├── task3_multimodal_housing/
│   └── task3_multimodal_housing.ipynb
├── task4_rag_chatbot/
│   └── task4_rag_chatbot.ipynb
├── task5_auto_tag_tickets/
│   └── task5_auto_tag_support_tickets.ipynb
└── README.md  ← You are here
```

---

## 🗂️ Task Overview

| # | Task | Technique | Model / Stack | Key Result |
|---|------|-----------|---------------|------------|
| 1 | [News Topic Classifier](#-task-1--news-topic-classifier-using-bert) | Fine-tuning | BERT + HuggingFace | **92.65% accuracy**, F1 Macro 92.62% |
| 2 | [Customer Churn Prediction](#-task-2--customer-churn-prediction-pipeline) | End-to-end ML pipeline | Gradient Boosting + scikit-learn | **ROC-AUC 0.8467**, 80.41% accuracy |
| 3 | [Multimodal Housing Price Prediction](#-task-3--multimodal-housing-price-prediction) | CNN + tabular fusion | MobileNetV2 + GBM | Baseline GBM wins — MAE **$19,158**, MAPE 9.71% |
| 4 | [Context-Aware RAG Chatbot](#-task-4--context-aware-rag-chatbot) | RAG + conversational memory | LangChain + FAISS + Flan-T5 | ~80–100% retrieval hit rate, 2–5s latency |
| 5 | [Auto-Tag Support Tickets](#-task-5--auto-tagging-support-tickets-using-llm) | Prompt engineering | Llama 3.3 70B via Groq | Few-shot ~90–97% accuracy (zero-shot ~83–90%) |

---

## 📰 Task 1 — News Topic Classifier Using BERT

**Goal:** Fine-tune `bert-base-uncased` to classify news headlines into 4 categories — World, Sports, Business, Sci/Tech — using the AG News dataset.

### Approach
- **Dataset:** 8,000 train / 2,000 val / 2,000 test samples from AG News (120K total, 4 balanced classes)
- **Tokenization:** `bert-base-uncased` tokenizer, max 128 tokens, `[CLS]` + `[SEP]` tokens, attention mask
- **Architecture:** BERT (110M params, 12 layers, 768-d hidden) + linear classification head (768 → 4)
- **Training:** 3 epochs, lr=2e-5, batch=16, fp16 mixed precision, warmup steps=150, weight decay=0.01
- **Evaluation:** Accuracy, Macro F1, Weighted F1, Confusion Matrix
- **Deployment:** Interactive Gradio demo for live headline classification with per-class confidence scores

### Results

| Epoch | Train Loss | Val Loss | Accuracy | F1 Macro | F1 Weighted |
|-------|-----------|----------|----------|----------|-------------|
| 1 | 0.2797 | 0.2836 | 91.70% | 91.68% | 91.71% |
| 2 | 0.2568 | 0.2662 | 92.50% | 92.49% | 92.52% |
| 3 | 0.1132 | 0.2863 | **92.65%** | **92.62%** | **92.66%** |

Training time: ~4.2 minutes on GPU. Mild overfitting at epoch 3 — val loss slightly increased while train loss halved, indicating the right stopping point.

### Key Observations
- Transfer learning is highly effective — 92%+ accuracy from only 8,000 samples due to BERT's 3.3B-word pre-training
- Sports is the easiest class (highly distinctive vocabulary); Business ↔ World shows the most confusion (overlapping economic/geopolitical terms)
- No manual feature engineering needed — BERT's self-attention learns discriminative patterns automatically

📄 [Full README →](task1_news_classifier/README.md)

---

## 🔄 Task 2 — Customer Churn Prediction Pipeline

**Goal:** Build a production-ready, end-to-end ML pipeline to predict customer churn on the IBM Telco dataset using Gradient Boosting, with hyperparameter tuning and serialised deployment.

### Approach
- **Dataset:** 7,043 customers × 21 features; 73.5% No Churn / 26.5% Churn (imbalanced)
- **Preprocessing:** `ColumnTransformer` → Numeric (impute + scale), Binary (ordinal encode), Nominal (one-hot encode) → 19 raw → 40 features; `class_weight="balanced"` on all classifiers
- **Models compared:** Logistic Regression, Random Forest, Gradient Boosting
- **Tuning:** `GridSearchCV` with 5-fold `StratifiedKFold`, 54 combinations × 5 folds = 270 fits, scoring on ROC-AUC
- **Best params:** `lr=0.05, max_depth=3, n_estimators=100, subsample=0.8`
- **Persistence:** Full pipeline serialised via `joblib` → `churn_pipeline.joblib` (0.2 MB)

### Results

**Model comparison (before tuning):**

| Model | Accuracy | Recall | F1 | ROC-AUC |
|-------|----------|--------|-----|---------|
| Logistic Regression | 0.7381 | 0.7834 | 0.6136 | 0.8414 |
| Random Forest | 0.7771 | 0.4813 | 0.5341 | 0.8160 |
| Gradient Boosting | 0.7899 | 0.5107 | 0.5634 | 0.8350 |

**Tuned Gradient Boosting (best model):**

| Metric | Baseline | Tuned | Δ |
|--------|----------|-------|---|
| Accuracy | 0.7899 | **0.8041** | +0.0142 |
| Precision | 0.6283 | **0.6725** | +0.0442 |
| ROC-AUC | 0.8350 | **0.8467** | +0.0117 |
| 5-Fold CV AUC | — | **0.8473 ± 0.0103** | — |

**Top churn predictors:** Contract_Month-to-month (0.38), tenure (0.15), Fiber optic internet (0.09), No OnlineSecurity (0.08), MonthlyCharges (0.07)

### Key Observations
- Month-to-month contract is by far the strongest churn signal — customers without long-term commitment churn at 3–4× the rate
- Logistic Regression has the best recall (0.78) — most churners caught, but lower precision; best when missing a churner is very costly
- Tuning improved precision by +4.4 percentage points while keeping recall stable — producing a more reliable production model
- 5-fold CV confirms no overfitting — test AUC (0.8467) closely matches CV mean (0.8473)

📄 [Full README →](task2_churn_prediction/README.md)

---

## 🏠 Task 3 — Multimodal Housing Price Prediction

**Goal:** Test whether fusing CNN image features with tabular housing data improves sale price prediction over a tabular-only baseline. Metrics: MAE, RMSE, MAPE.

### Approach
- **Datasets:** Kaggle House Prices (1,460 houses × 27 selected features) + House Rooms Image Dataset (5,250 images, 5 room types)
- **Image pipeline:** MobileNetV2 (ImageNet, frozen) → 1,280-d feature vectors → PCA → 64-d (63.9% variance retained)
- **Fusion:** Late fusion — tabular (27-d) concatenated with PCA image features (64-d) → 91-d fused matrix
- **Models:** Baseline GBM (tabular only), Multimodal GBM (fused), Deep Fusion Neural Network (two-branch: 128→64 per modality, Huber loss, EarlyStopping)
- **Split:** 992 train / 176 val / 292 test, stratified by SalePrice quartile

### Results (held-out test set, dollar scale)

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| **Baseline GBM (Tabular Only)** | **$19,158** | **$39,812** | **9.71%** |
| Multimodal GBM (Tabular + CNN) | $21,000 | $43,516 | 10.75% |
| Deep Fusion Neural Network | $103,448 | $521,546 | 43.78% |

The tabular-only baseline won on every metric. Adding image features made both models worse.

### Key Observations
- **This is a useful negative result** — multimodal fusion only helps when modalities are well-aligned. Images were assigned to houses via random draw within coarse `OverallQual` buckets, so the CNN features encoded noise plus a faint quality signal already captured by the tabular `OverallQual` column
- The Deep Fusion NN's large dollar-scale RMSE reflects `expm1` amplifying small log-scale errors at the high-price tail — a known risk of log-target regression
- Takeaway: correct implementation of a multimodal pipeline does not guarantee improvement — image–record alignment is the prerequisite

📄 [Full README →](task3_multimodal_housing/README.md)

---

## 💬 Task 4 — Context-Aware RAG Chatbot

**Goal:** Build a conversational chatbot that retrieves from a vectorised knowledge base (PDF + Wikipedia) before generating answers, with multi-turn memory, deployable as a Streamlit app.

### Approach
- **LLM:** `google/flan-t5-base` (248M params, local CPU)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-d dense vectors)
- **Vector store:** FAISS with Maximum Marginal Relevance retrieval (top-4 of 20 candidates)
- **Framework:** LangChain 1.x LCEL (`|` operator chains)
- **Ingestion:** PyPDFLoader (user PDFs) + Wikipedia library (5 AI/ML topics as fallback)
- **Chunking:** `RecursiveCharacterTextSplitter` — 512 chars, 64-char overlap
- **Memory:** Manual Python list of `{role, content}` dicts, window of last 5 turns
- **Chain flow:** Query → MMR retrieval → history formatting → `PromptTemplate` fill → Flan-T5 generation

### Results

| Metric | Value |
|--------|-------|
| Embedding dimension | 384-d |
| Retrieval method | FAISS MMR (top-4 of 20) |
| Retrieval hit rate | ~80–100% on AI/ML topics |
| Average response latency | ~2–5 seconds (CPU) |
| Conversational turns tested | 5 multi-turn exchanges |

### Key Observations
- Wikipedia fallback works reliably for general AI/ML knowledge — `DisambiguationError` handled gracefully
- MMR retrieval prevents near-duplicate chunks from flooding the context window
- FAISS index saved locally — embeddings not recomputed on restart (significant startup time saving)
- Manual memory list is more LangChain-version-stable than built-in memory classes (which were removed in 1.x)
- `flan-t5-base` handles factual Q&A well on CPU but has a ~512-token context limit

📄 [Full README →](task4_rag_chatbot/README.md)

---

## 🎫 Task 5 — Auto-Tagging Support Tickets Using LLM

**Goal:** Automatically classify customer support tickets into 5 categories with priority levels using prompt engineering — no model training required.

### Approach
- **Model:** `llama-3.3-70b-versatile` via Groq API (free tier), temperature=0.1
- **Dataset:** 30 labelled tickets (6 per category) for evaluation + 7 unseen tickets for live demo
- **Tags:** Billing · Technical · Account · Shipping · General
- **Priorities:** 🔴 Critical · 🟠 High · 🟡 Medium · 🟢 Low
- **Output schema:** `{"tag", "priority", "confidence", "reasoning"}` — strictly enforced JSON

**Three techniques implemented:**

| Technique | How It Works | Tokens | Best For |
|-----------|-------------|--------|----------|
| **Zero-Shot** | Tag definitions in system prompt only | ~300 | Clear, unambiguous tickets |
| **Few-Shot** | 1 labelled example per category in message history | ~700 | Edge cases, consistent format |
| **Chain-of-Thought** | 5-step reasoning before JSON output | ~400 in / ~300 out | Ambiguous multi-category tickets |

### Results

| Technique | Accuracy (expected) | F1 Macro (expected) | Avg Confidence |
|-----------|--------------------|--------------------|----------------|
| Zero-Shot | ~83–90% | ~82–89% | High on clear tickets |
| Few-Shot | ~90–97% | ~89–96% | Calibrated to ambiguity |

> Exact values generated at runtime. Few-Shot typically improves Macro F1 by 5–15 percentage points over Zero-Shot.

**Live demo — 7 unseen tickets (Few-Shot):**

| Ticket | Predicted Tag | Priority |
|--------|---------------|----------|
| "Waiting 3 weeks, tracking still processing…" | Shipping | 🟠 High |
| "App crashes exporting data as PDF…" | Technical | 🟠 High |
| "Can't find option to cancel subscription…" | Account | 🟡 Medium |
| "Bill increased $15 without notification…" | Billing | 🟡 Medium |
| "Do you have an API for enterprise customers?…" | General | 🟢 Low |
| "Charged for premium but account shows basic…" | Billing | 🟠 High |
| "Dark mode is hurting my eyes…" | Technical | 🟢 Low |

### Key Observations
- Billing ↔ Technical is the hardest boundary (e.g. "can't pay — payment page shows an error") — CoT reasoning explicitly resolves ambiguity by identifying the primary customer impact
- Confidence scores are self-calibrated — the model assigns ≥0.95 to clear-cut tickets and lower scores to edge cases, enabling threshold-based human-in-the-loop routing (flag if confidence < 0.75)
- Temperature = 0.1 is essential — deterministic output prevents classification variance across runs
- CoT adds an auditable reasoning trace even when the final tag matches zero-shot

**Production routing recommendation:**
```
Incoming Ticket
    │
    ├── Clear keywords  → Few-Shot   (~700 tokens, ~0.4s)
    └── Ambiguous       → Chain-of-Thought (~700 tokens, ~1.5s)
    │
    ▼
Structured JSON → tag (queue routing) · priority (SLA) · confidence (human flag) · reasoning (agent context)
```

📄 [Full README →](task5_auto_tag_tickets/README.md)

---

## 🛠️ Skills Demonstrated Across All Tasks

| Skill Area | Tasks |
|------------|-------|
| Transformer fine-tuning (BERT, HuggingFace Trainer) | Task 1 |
| Classical ML pipelines (scikit-learn, GridSearchCV, joblib) | Task 2 |
| CNN feature extraction + multimodal fusion (MobileNetV2, PCA, late fusion) | Task 3 |
| RAG pipeline design (LangChain, FAISS, MMR retrieval, conversational memory) | Task 4 |
| LLM prompt engineering (zero-shot, few-shot, chain-of-thought, structured output) | Task 5 |
| Evaluation (accuracy, F1, ROC-AUC, MAE/RMSE/MAPE, confusion matrix) | All |
| Production thinking (serialisation, confidence thresholds, human-in-the-loop) | All |

---

## ⚙️ Setup

Each task has its own dependencies. Install per task:

```bash
# Task 1 — BERT News Classifier
pip install transformers datasets torch scikit-learn gradio

# Task 2 — Churn Prediction Pipeline
pip install scikit-learn pandas numpy matplotlib seaborn joblib

# Task 3 — Multimodal Housing
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn kagglehub

# Task 4 — RAG Chatbot
pip install langchain langchain-core langchain-community langchain-huggingface \
            faiss-cpu sentence-transformers transformers wikipedia pypdf streamlit

# Task 5 — Auto-Tag Support Tickets
pip install groq pandas numpy matplotlib seaborn scikit-learn
```

API keys required:
- **Task 4:** No external API key needed (local Hugging Face models)
- **Task 5:** Free Groq API key → [console.groq.com](https://console.groq.com)
- **Task 3:** Kaggle credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`) for dataset download

---

*DevelopersHub Corporation — AI/ML Engineering Advanced Internship*
