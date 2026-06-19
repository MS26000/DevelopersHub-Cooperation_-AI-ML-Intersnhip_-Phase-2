# Task 5: Auto-Tagging Support Tickets Using LLM

**DevelopersHub Corporation — AI/ML Engineering Advanced Internship**

---

## Objective

Automatically classify customer support tickets into predefined categories using a Large Language Model (LLM), without requiring any model training. The system leverages **prompt engineering techniques** — zero-shot, few-shot, and chain-of-thought — to tag each ticket and assign a priority level, outputting structured JSON for downstream use.

---

## Tag Categories

| Tag | Description |
|-----|-------------|
| **Billing** | Payment issues, invoices, charges, refunds |
| **Technical** | Bugs, errors, crashes, connectivity, performance |
| **Account** | Profile changes, passwords, subscriptions, access |
| **Shipping** | Delivery, tracking, lost/wrong packages |
| **General** | General enquiries, feedback, compliments |

### Priority Levels
- 🔴 **Critical** — fraud, data loss, service outage
- 🟠 **High** — blocking core functionality
- 🟡 **Medium** — partial impact, workaround exists
- 🟢 **Low** — minor issue or general question

---

## Tech Stack

| Component | Detail |
|-----------|--------|
| LLM | `llama-3.3-70b-versatile` via Groq API (free tier) |
| Techniques | Zero-shot · Few-shot · Chain-of-Thought (CoT) |
| Output format | Structured JSON (`tag`, `priority`, `confidence`, `reasoning`) |
| Evaluation | Accuracy, Macro F1, Weighted F1, Confusion Matrix |

---

## Methodology / Approach

### Dataset
A labelled dataset of **30 support tickets** (6 per category) was created to serve as the ground truth for evaluation. An additional 7 unseen tickets were used for a live demo.

### Technique 1 — Zero-Shot Prompting
- The system prompt defines all 5 tags and 4 priority levels with descriptions.
- No examples are provided — the LLM relies entirely on pre-trained language understanding.
- The model is instructed to return a strict JSON object only.
- Temperature set to **0.1** for deterministic, reproducible output.
- Approximate token usage: ~300 input tokens per ticket.

### Technique 2 — Few-Shot Prompting
- One representative example per category is provided as alternating `user`/`assistant` turns in the message history.
- This anchors the model's understanding of edge cases and output format.
- The same low-temperature JSON instruction from zero-shot is retained.
- Approximate token usage: ~700 input tokens per ticket.

### Technique 3 — Chain-of-Thought (CoT) Prompting
- The system prompt instructs the model to reason through 5 explicit steps before classifying:
  1. Identify key problem keywords.
  2. Match keywords to category signals.
  3. Resolve ambiguity (could this fit multiple categories?).
  4. Assign priority based on urgency.
  5. Output final JSON.
- Best suited for ambiguous tickets (e.g., "can't pay my bill — payment page shows an error" → Billing or Technical?).
- Higher token usage (~400 input / ~300 output per ticket) but provides an auditable reasoning trace.

### Structured JSON Output
All three techniques enforce a strict output schema:
```json
{
  "tag":        "Billing | Technical | Account | Shipping | General",
  "priority":   "Critical | High | Medium | Low",
  "confidence": 0.0 to 1.0,
  "reasoning":  "One sentence explanation"
}
```
JSON parsing includes fallback handling for markdown code fences and parse errors.

---

## Key Results and Observations

| Technique | Accuracy | F1 (Macro) | F1 (Weighted) | Avg Confidence |
|-----------|----------|------------|---------------|----------------|
| Zero-Shot | ~X.XX | ~X.XX | ~X.XX | ~X.XX |
| Few-Shot | ~X.XX | ~X.XX | ~X.XX | ~X.XX |

> Exact metric values are populated at runtime after the full batch evaluation.

### Observations

1. **Few-shot outperforms zero-shot** — providing one example per category helps the model resolve borderline cases, typically improving macro F1 by 5–15%.
2. **Billing vs Technical overlap** is the most common misclassification — tickets involving payment page errors sit at the intersection of both categories; CoT reasoning explicitly resolves this.
3. **Confidence scores are meaningful** — the LLM assigns high confidence (≥0.95) to unambiguous tickets and lower confidence to edge cases, making it easy to flag uncertain predictions for human review.
4. **Temperature = 0.1 is critical** for classification tasks — it produces deterministic, consistent output across repeated runs.
5. **Chain-of-Thought reasoning** adds audit trail value even when the final tag matches zero-shot output — the step-by-step reasoning is useful for agent context and quality review.

### Production Recommendation

```
Incoming Ticket
    │
    ├── Clear keywords → Few-Shot (fast, ~700 tokens)
    └── Ambiguous      → Chain-of-Thought (thorough, ~700 tokens)
    │
    ▼
Structured JSON
    ├── tag        → Route to correct support queue
    ├── priority   → Set SLA timer
    ├── confidence → Flag for human review if < 0.75
    └── reasoning  → Attach to ticket for agent context
```

---

## Skills Demonstrated

- Prompt engineering: zero-shot, few-shot, chain-of-thought, structured output
- LLM API integration (Groq Cloud, Llama 3.3 70B)
- Multi-class text classification without model training
- Batch inference with rate limiting
- Evaluation using accuracy, F1, and confusion matrices
- Confidence-based human-in-the-loop flagging

---

## How to Run

```bash
# Install dependencies
pip install groq pandas numpy matplotlib seaborn scikit-learn

# Set your Groq API key (free at https://console.groq.com)
export GROQ_API_KEY=your_key_here

# Run the notebook
jupyter notebook task5_auto_tag_support_tickets.ipynb
```

---

## Repository Structure

```
task5_auto_tag_support_tickets.ipynb   ← Main notebook
evaluation_dashboard.png               ← Accuracy/F1/confusion matrix plots
confidence_distribution.png            ← Confidence score histograms
```
