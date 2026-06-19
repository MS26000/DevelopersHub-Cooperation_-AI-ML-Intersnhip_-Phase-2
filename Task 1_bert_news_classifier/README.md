# 📰 Task 1: News Topic Classifier Using BERT

**DevelopersHub Corporation – AI/ML Engineering Advanced Internship**

---

## 🎯 Objective

The goal of this task is to fine-tune a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model to automatically classify news headlines into one of four topic categories:

- 🌍 World
- 🏅 Sports
- 💼 Business
- 💻 Sci/Tech

The model is trained on the **AG News Dataset**, a widely used benchmark for text classification containing over 120,000 news articles collected from more than 2,000 news sources.

---

## 🧠 Methodology / Approach

### 1. Dataset
- **Source:** AG News (`fancyzhx/ag_news` via HuggingFace Datasets)
- **Total size:** 120,000 training samples / 7,600 test samples — perfectly balanced at 25% per class
- **Subset used:** 8,000 training / 2,000 validation / 2,000 test samples for faster experimentation

### 2. Tokenization & Preprocessing
- Used `bert-base-uncased` tokenizer (vocab size: 30,522)
- Each headline is tokenized with:
  - `[CLS]` token prepended (used for classification)
  - `[SEP]` token appended
  - Padding / truncation to a fixed length of **128 tokens**
  - Attention mask to distinguish real tokens from padding

### 3. Model Architecture
- **Base model:** `bert-base-uncased` (110M parameters, 12 layers, 12 attention heads, 768 hidden dimensions)
- **Classification head:** Linear layer added on top (768 → 4 output classes)
- The pre-trained BERT backbone is fine-tuned end-to-end at a low learning rate; the classification head is trained from scratch

### 4. Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Warmup Steps | 150 (10% of total steps) |
| Weight Decay | 0.01 |
| Max Token Length | 128 |
| Mixed Precision | fp16 (GPU) |

- **Framework:** HuggingFace `Trainer` API with `DataCollatorWithPadding`
- **Best model selection:** based on macro F1-score across epochs

### 5. Evaluation Metrics
- **Accuracy** — overall percentage of correct predictions
- **F1 Macro** — unweighted mean F1 across all 4 classes (treats each class equally)
- **F1 Weighted** — F1 weighted by class support
- **Confusion Matrix** — both raw counts and normalised percentages

### 6. Deployment
- Interactive demo built with **Gradio** running inline inside the notebook
- Users can type any news headline and receive an instant topic prediction with confidence scores for all 4 classes
- Inference pipeline loaded from the saved fine-tuned model using HuggingFace `pipeline`

---

## 📊 Key Results & Observations

### Training Results

| Epoch | Train Loss | Val Loss | Accuracy | F1 Macro | F1 Weighted |
|-------|-----------|----------|----------|----------|-------------|
| 1 | 0.2797 | 0.2836 | 91.70% | 91.68% | 91.71% |
| 2 | 0.2568 | 0.2662 | 92.50% | 92.49% | 92.52% |
| 3 | 0.1132 | 0.2863 | **92.65%** | **92.62%** | **92.66%** |

- **Training time:** ~4.2 minutes on GPU (CUDA)
- **Final test accuracy: 92.65%**
- **Final macro F1: 92.62%**

### Key Observations

1. **Transfer learning is highly effective** — BERT achieves over 92% accuracy on only 8,000 training samples because its pre-training on 3.3B words gives it a deep understanding of English language structure before fine-tuning even begins.

2. **Mild overfitting at epoch 3** — training loss dropped sharply to 0.113 while validation loss slightly increased (0.286 vs 0.266 at epoch 2), indicating the model was approaching its limit on this subset size. Three epochs was the right stopping point.

3. **Sports is the easiest class to classify** — sports vocabulary (goals, matches, players, championships) is highly distinctive and does not overlap significantly with the other three categories.

4. **Business and World show the most confusion** — economic news and geopolitical news share vocabulary (e.g. trade, sanctions, government, economy), making these two the hardest to separate.

5. **Sci/Tech is well separated** — technology-specific terms (AI, chip, device, software, launch) are distinctive enough for BERT to classify with high confidence.

6. **No manual feature engineering needed** — BERT's self-attention mechanism automatically learns which words and phrases are most discriminative for each category.

---

## 💾 Saved Outputs

| File | Description |
|------|-------------|
| `bert_news_classifier/` | Fine-tuned model weights + tokenizer |
| `label_config.json` | Label-to-ID mapping |
| `class_distribution.png` | Train/test class balance chart |
| `confusion_matrix.png` | Raw and normalised confusion matrices |
| `per_class_performance.png` | Per-class F1 and accuracy bar charts |

---

## 🔭 Future Improvements

- Train on the full 120,000 samples for higher accuracy (~97-98% expected)
- Try **DistilBERT** for 40% faster inference with minimal accuracy loss
- Use **RoBERTa** or **DeBERTa** for potentially higher F1
- Add confidence thresholding to reject low-confidence predictions
- Deploy to **HuggingFace Spaces** for public access
- Extend to more news categories using a larger dataset
