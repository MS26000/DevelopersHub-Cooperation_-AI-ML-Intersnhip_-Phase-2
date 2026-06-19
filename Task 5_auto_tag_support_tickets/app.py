"""
🎫 Auto-Tag Support Tickets — Streamlit App
DevelopersHub Corporation – AI/ML Engineering Advanced Internship (Task 5)

Classifies customer support tickets into Billing / Technical / Account /
Shipping / General using an LLM (Groq's Llama-3.3-70B) via Zero-Shot,
Few-Shot, or Chain-of-Thought prompting — no training required.
"""

import json
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto-Tag Support Tickets",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"
TAGS = ["Billing", "Technical", "Account", "Shipping", "General"]
PRIORITIES = ["Critical", "High", "Medium", "Low"]
PRIORITY_ICON = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
TAG_COLOR = {
    "Billing": "#58a6ff",
    "Technical": "#f78166",
    "Account": "#3fb950",
    "Shipping": "#d2a8ff",
    "General": "#ffa657",
}

TICKETS = [
    {"id": "T001", "ticket": "I was charged twice for my subscription this month. Please refund the extra payment immediately.", "label": "Billing"},
    {"id": "T002", "ticket": "My invoice shows an incorrect amount. I should have been billed $29 but was charged $59.", "label": "Billing"},
    {"id": "T003", "ticket": "I cancelled my plan 2 weeks ago but I'm still being charged. This is unacceptable.", "label": "Billing"},
    {"id": "T004", "ticket": "Can you explain the charges on my latest bill? I see a fee I don't recognise.", "label": "Billing"},
    {"id": "T005", "ticket": "How do I update my credit card information for automatic payments?", "label": "Billing"},
    {"id": "T006", "ticket": "I need a receipt for my last three payments for tax purposes.", "label": "Billing"},
    {"id": "T007", "ticket": "The app keeps crashing every time I try to open it on my iPhone. I've tried reinstalling.", "label": "Technical"},
    {"id": "T008", "ticket": "I cannot log into my account. It says invalid password but I'm using the correct one.", "label": "Technical"},
    {"id": "T009", "ticket": "The website is extremely slow and pages take over 30 seconds to load.", "label": "Technical"},
    {"id": "T010", "ticket": "I'm getting a 500 internal server error whenever I try to submit a form.", "label": "Technical"},
    {"id": "T011", "ticket": "My dashboard data hasn't updated in 3 days. The charts show old information.", "label": "Technical"},
    {"id": "T012", "ticket": "The video call feature is not working. No audio comes through on my end.", "label": "Technical"},
    {"id": "T013", "ticket": "I can't upload files larger than 5MB even though my plan allows 50MB.", "label": "Technical"},
    {"id": "T014", "ticket": "I need to change the email address associated with my account.", "label": "Account"},
    {"id": "T015", "ticket": "I forgot my password and the reset email is not arriving in my inbox.", "label": "Account"},
    {"id": "T016", "ticket": "Please delete my account and all associated data per GDPR regulations.", "label": "Account"},
    {"id": "T017", "ticket": "I want to upgrade my plan from Basic to Premium. How do I do this?", "label": "Account"},
    {"id": "T018", "ticket": "How do I add a second user to my account? I want to share access with my colleague.", "label": "Account"},
    {"id": "T019", "ticket": "My account has been locked after too many failed login attempts. Please unlock it.", "label": "Account"},
    {"id": "T020", "ticket": "My order was supposed to arrive 5 days ago and tracking shows it's still in transit.", "label": "Shipping"},
    {"id": "T021", "ticket": "I received the wrong item in my delivery. I ordered the blue model but got red.", "label": "Shipping"},
    {"id": "T022", "ticket": "My package shows as delivered but I never received it. It may have been stolen.", "label": "Shipping"},
    {"id": "T023", "ticket": "I need to change my delivery address. The order was placed 2 hours ago.", "label": "Shipping"},
    {"id": "T024", "ticket": "Can you provide a tracking number for my order placed last Tuesday?", "label": "Shipping"},
    {"id": "T025", "ticket": "One item from my order of 3 products is missing from the package.", "label": "Shipping"},
    {"id": "T026", "ticket": "What are your customer support hours? I need to speak with someone urgently.", "label": "General"},
    {"id": "T027", "ticket": "I just wanted to say your team was incredibly helpful last week. Great service!", "label": "General"},
    {"id": "T028", "ticket": "Do you offer student discounts or non-profit pricing?", "label": "General"},
    {"id": "T029", "ticket": "I have a suggestion to improve your mobile app — would you like to hear it?", "label": "General"},
    {"id": "T030", "ticket": "Are your services available in Pakistan? I'm interested in signing up.", "label": "General"},
]

ZERO_SHOT_SYSTEM = """You are an expert customer support ticket classifier.

Your task is to analyse support tickets and assign the most appropriate tag.

Available tags:
- Billing   : Payment issues, invoices, charges, refunds, pricing
- Technical : Bugs, errors, crashes, connectivity, performance problems
- Account   : Profile changes, passwords, subscriptions, access management
- Shipping  : Delivery, tracking, lost/wrong packages, address changes
- General   : General enquiries, feedback, compliments, policy questions

Also assign a priority level:
- Critical : Service down, data loss, billing fraud, security breach
- High     : Significant impact blocking core functionality
- Medium   : Partial impact, workaround available
- Low      : Minor issue or general question

Return ONLY a valid JSON object with this exact structure:
{
  "tag"       : "<one of: Billing|Technical|Account|Shipping|General>",
  "priority"  : "<one of: Critical|High|Medium|Low>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning" : "<one sentence explaining why>"
}"""

FEW_SHOT_EXAMPLES = [
    {
        "ticket": "I was charged twice this month and need an immediate refund.",
        "output": '{"tag":"Billing","priority":"High","confidence":0.98,"reasoning":"Customer reports duplicate charge and requests refund — clear billing issue."}',
    },
    {
        "ticket": "The dashboard keeps showing a blank screen after login.",
        "output": '{"tag":"Technical","priority":"High","confidence":0.95,"reasoning":"Blank screen after login indicates a technical bug affecting core functionality."}',
    },
    {
        "ticket": "I need to update my email address on the account.",
        "output": '{"tag":"Account","priority":"Low","confidence":0.97,"reasoning":"Email address change is an account management request."}',
    },
    {
        "ticket": "My package shows delivered but I never received it.",
        "output": '{"tag":"Shipping","priority":"High","confidence":0.96,"reasoning":"Missing delivered package is a shipping and delivery issue."}',
    },
    {
        "ticket": "Do you have a mobile app available for Android?",
        "output": '{"tag":"General","priority":"Low","confidence":0.94,"reasoning":"Product availability question — general enquiry, no specific problem."}',
    },
]

COT_SYSTEM = """You are an expert customer support ticket classifier.

When classifying a ticket, think step by step:

STEP 1 — KEYWORD ANALYSIS: What are the key problem words in this ticket?
STEP 2 — CATEGORY SIGNALS: Which category do these keywords point to?
  - Billing   : charge, payment, invoice, refund, bill, price, fee, subscription cost
  - Technical : error, crash, bug, not working, slow, broken, fails, cannot, 500
  - Account   : password, email, profile, login, subscription plan, delete account
  - Shipping  : delivery, tracking, package, order, arrived, shipped, address
  - General   : question, hours, discount, feedback, suggestion, availability
STEP 3 — AMBIGUITY CHECK: Could this fit multiple categories? Which is primary?
STEP 4 — PRIORITY: How urgently does this need resolution?
  - Critical : fraud, data loss, complete service failure
  - High     : blocking core use, missing payment, wrong delivery
  - Medium   : partial feature broken, delay in non-critical feature
  - Low      : question, minor issue, enhancement request
STEP 5 — FINAL ANSWER: Output JSON only.

Return your full reasoning followed by the JSON. Format:
REASONING: <your step-by-step thinking>
JSON: {"tag":"...","priority":"...","confidence":0.0,"reasoning":"<one sentence>"}"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def _strip_fences(raw: str) -> str:
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def classify_zero_shot(client: Groq, ticket_text: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ZERO_SHOT_SYSTEM},
            {"role": "user", "content": f'Classify this ticket:\n\n"{ticket_text}"'},
        ],
        max_tokens=200,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return {"tag": "General", "priority": "Low", "confidence": 0.0, "reasoning": "Parse error"}


def classify_few_shot(client: Groq, ticket_text: str) -> dict:
    messages = [{"role": "system", "content": ZERO_SHOT_SYSTEM}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f'Classify this ticket:\n\n"{ex["ticket"]}"'})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": f'Classify this ticket:\n\n"{ticket_text}"'})

    response = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=200, temperature=0.1
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return {"tag": "General", "priority": "Low", "confidence": 0.0, "reasoning": "Parse error"}


def classify_cot(client: Groq, ticket_text: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": f'Classify this support ticket step by step:\n\n"{ticket_text}"'},
        ],
        max_tokens=400,
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
    try:
        if "JSON:" in raw:
            json_part = raw.split("JSON:")[-1].strip()
        elif "```" in raw:
            json_part = _strip_fences(raw)
        else:
            json_part = raw
        result = json.loads(json_part.strip())
        result["_reasoning_steps"] = raw.split("JSON:")[0].strip() if "JSON:" in raw else ""
        return result
    except json.JSONDecodeError:
        return {
            "tag": "General",
            "priority": "Low",
            "confidence": 0.0,
            "reasoning": "Parse error",
            "_reasoning_steps": raw,
        }


CLASSIFIERS = {
    "Zero-Shot": classify_zero_shot,
    "Few-Shot": classify_few_shot,
    "Chain-of-Thought": classify_cot,
}


def compute_metrics(df: pd.DataFrame, name: str) -> dict:
    acc = accuracy_score(df["true_label"], df["pred_label"])
    f1m = f1_score(df["true_label"], df["pred_label"], average="macro", zero_division=0)
    f1w = f1_score(df["true_label"], df["pred_label"], average="weighted", zero_division=0)
    return {
        "Technique": name,
        "Accuracy": round(acc, 4),
        "F1 Macro": round(f1m, 4),
        "F1 Weighted": round(f1w, 4),
        "Avg Confidence": round(df["confidence"].mean(), 4),
        "Correct": int(df["correct"].sum()),
        "Total": len(df),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: ticket, technique, result
if "eval_results" not in st.session_state:
    st.session_state.eval_results = {}  # technique -> dataframe

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎫 Auto-Tag Tickets")
    st.caption("DevelopersHub Corp · Task 5 · AI/ML Advanced Internship")

    st.markdown("### 🔑 Groq API Key")
    api_key = st.text_input(
        "Enter your Groq API key",
        type="password",
        help="Get a free key at console.groq.com — never hard-code keys in shared apps.",
    )
    st.caption(f"Model: `{MODEL}`")

    st.divider()
    st.markdown("### 🧠 Prompting Technique")
    technique = st.radio(
        "Choose technique",
        list(CLASSIFIERS.keys()),
        index=1,
        help=(
            "Zero-Shot: tag definitions only.\n"
            "Few-Shot: 1 labelled example per category.\n"
            "Chain-of-Thought: model reasons step-by-step before answering."
        ),
    )

    st.divider()
    st.markdown("### 📚 Tags & Priorities")
    st.write(", ".join(TAGS))
    st.write(", ".join(f"{PRIORITY_ICON[p]} {p}" for p in PRIORITIES))

    st.divider()
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("🎫 Auto-Tag Support Tickets Using LLM")
st.markdown(
    "Automatically classify customer support tickets into **Billing · Technical · "
    "Account · Shipping · General** using **Zero-Shot, Few-Shot, or Chain-of-Thought** "
    "prompting — powered by Llama-3.3-70B via Groq. No model training required."
)

if not api_key:
    st.info("👈 Enter your Groq API key in the sidebar to get started. Get a free key at [console.groq.com](https://console.groq.com).")

tab_classify, tab_batch, tab_eval, tab_about = st.tabs(
    ["🔮 Classify a Ticket", "📦 Batch Tag Tickets", "📊 Technique Evaluation", "ℹ️ About"]
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Single ticket classification
# ──────────────────────────────────────────────────────────────────────────────
with tab_classify:
    st.subheader("Classify a single ticket")

    sample_options = ["— Type your own —"] + [f"{t['id']}: {t['ticket'][:60]}..." for t in TICKETS]
    choice = st.selectbox("Optionally pick a sample ticket", sample_options)

    default_text = ""
    if choice != "— Type your own —":
        idx = sample_options.index(choice) - 1
        default_text = TICKETS[idx]["ticket"]

    ticket_text = st.text_area(
        "Support ticket text",
        value=default_text,
        height=120,
        placeholder="e.g. I was charged twice this month and need a refund...",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        classify_clicked = st.button("🔮 Classify", type="primary", use_container_width=True)

    if classify_clicked:
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        elif not ticket_text.strip():
            st.warning("Please enter a ticket or pick a sample.")
        else:
            with st.spinner(f"Classifying with {technique}..."):
                try:
                    client = get_client(api_key)
                    result = CLASSIFIERS[technique](client, ticket_text)
                    st.session_state.history.insert(
                        0, {"ticket": ticket_text, "technique": technique, "result": result}
                    )
                except Exception as e:
                    st.error(f"Error calling Groq API: {e}")
                    result = None

            if result:
                tag = result.get("tag", "General")
                priority = result.get("priority", "Low")
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "")

                c1, c2, c3 = st.columns(3)
                c1.metric("🏷️ Tag", tag)
                c2.metric(f"{PRIORITY_ICON.get(priority, '🟢')} Priority", priority)
                c3.metric("📊 Confidence", f"{confidence:.0%}")

                st.markdown(f"**💬 Reasoning:** {reasoning}")

                if result.get("_reasoning_steps"):
                    with st.expander("🧠 Show full chain-of-thought reasoning"):
                        st.text(result["_reasoning_steps"])

    if st.session_state.history:
        st.divider()
        st.subheader("🕘 Recent classifications")
        for item in st.session_state.history[:10]:
            r = item["result"]
            tag = r.get("tag", "General")
            icon = PRIORITY_ICON.get(r.get("priority", "Low"), "🟢")
            with st.container(border=True):
                st.markdown(f"**{item['ticket'][:100]}**")
                st.caption(
                    f"Technique: {item['technique']} · Tag: **{tag}** · "
                    f"Priority: {icon} {r.get('priority')} · "
                    f"Confidence: {r.get('confidence', 0):.0%}"
                )

# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Batch tagging
# ──────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("Tag multiple tickets at once")
    st.caption("Paste one ticket per line, or use the 30-ticket labelled demo dataset.")

    use_demo = st.checkbox("Use built-in 30-ticket demo dataset", value=False)

    if use_demo:
        batch_text = "\n".join(t["ticket"] for t in TICKETS)
        st.text_area("Tickets (preview)", value=batch_text, height=150, disabled=True)
    else:
        batch_text = st.text_area(
            "Tickets — one per line",
            height=180,
            placeholder=(
                "I've been waiting 3 weeks for my order...\n"
                "Your app crashes every time I try to export...\n"
                "Why did my bill increase by $15 this month..."
            ),
        )

    if st.button("📦 Run batch tagging", type="primary"):
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            lines = [l.strip() for l in batch_text.split("\n") if l.strip()]
            if not lines:
                st.warning("Please provide at least one ticket.")
            else:
                client = get_client(api_key)
                progress = st.progress(0.0, text="Starting...")
                rows = []
                for i, line in enumerate(lines):
                    try:
                        result = CLASSIFIERS[technique](client, line)
                    except Exception as e:
                        result = {"tag": "General", "priority": "Low", "confidence": 0.0, "reasoning": str(e)}
                    rows.append(
                        {
                            "Ticket": line[:80] + ("..." if len(line) > 80 else ""),
                            "Tag": result.get("tag", "General"),
                            "Priority": result.get("priority", "Low"),
                            "Confidence": result.get("confidence", 0.0),
                            "Reasoning": result.get("reasoning", ""),
                        }
                    )
                    progress.progress((i + 1) / len(lines), text=f"Classified {i+1}/{len(lines)}")
                    time.sleep(0.3)
                progress.empty()

                df_batch = pd.DataFrame(rows)
                st.success(f"Tagged {len(df_batch)} tickets using {technique}.")
                st.dataframe(
                    df_batch.style.format({"Confidence": "{:.0%}"}),
                    use_container_width=True,
                    hide_index=True,
                )

                fig = px.histogram(
                    df_batch, x="Tag", color="Tag",
                    color_discrete_map=TAG_COLOR,
                    title="Tag distribution for this batch",
                    category_orders={"Tag": TAGS},
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                csv = df_batch.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download results as CSV", csv, "tagged_tickets.csv", "text/csv")

# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — Technique evaluation on labelled dataset
# ──────────────────────────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Evaluate prompting techniques on the labelled dataset")
    st.caption(f"Runs all {len(TICKETS)} labelled tickets against the selected technique(s) and reports accuracy, F1, and a confusion matrix.")

    eval_techniques = st.multiselect(
        "Techniques to evaluate", list(CLASSIFIERS.keys()), default=["Zero-Shot", "Few-Shot"]
    )

    if st.button("📊 Run evaluation", type="primary"):
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        elif not eval_techniques:
            st.warning("Pick at least one technique.")
        else:
            client = get_client(api_key)
            for tname in eval_techniques:
                progress = st.progress(0.0, text=f"Running {tname}...")
                rows = []
                for i, t in enumerate(TICKETS):
                    try:
                        pred = CLASSIFIERS[tname](client, t["ticket"])
                        tag = pred.get("tag", "General")
                        conf = pred.get("confidence", 0.0)
                        prio = pred.get("priority", "Low")
                        reason = pred.get("reasoning", "")
                    except Exception as e:
                        tag, conf, prio, reason = "General", 0.0, "Low", str(e)
                    rows.append(
                        {
                            "id": t["id"],
                            "ticket": t["ticket"][:60] + "...",
                            "true_label": t["label"],
                            "pred_label": tag,
                            "confidence": conf,
                            "priority": prio,
                            "reasoning": reason,
                            "correct": tag == t["label"],
                        }
                    )
                    progress.progress((i + 1) / len(TICKETS), text=f"{tname}: {i+1}/{len(TICKETS)}")
                    time.sleep(0.3)
                progress.empty()
                st.session_state.eval_results[tname] = pd.DataFrame(rows)
            st.success(f"Evaluation complete for: {', '.join(eval_techniques)}")

    if st.session_state.eval_results:
        st.divider()
        st.markdown("### 📈 Metrics comparison")
        metrics_rows = [
            compute_metrics(df, name) for name, df in st.session_state.eval_results.items()
        ]
        metrics_df = pd.DataFrame(metrics_rows).set_index("Technique")
        st.dataframe(metrics_df, use_container_width=True)

        best = metrics_df["F1 Macro"].idxmax()
        st.markdown(f"🏆 **Best technique by F1 Macro: {best}**")

        fig = go.Figure()
        for name in metrics_df.index:
            fig.add_trace(
                go.Bar(
                    name=name,
                    x=["Accuracy", "F1 Macro", "F1 Weighted"],
                    y=metrics_df.loc[name, ["Accuracy", "F1 Macro", "F1 Weighted"]].values,
                )
            )
        fig.update_layout(barmode="group", title="Accuracy & F1 by technique", yaxis_range=[0, 1.1])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🔢 Confusion matrices")
        cols = st.columns(len(st.session_state.eval_results))
        for col, (name, df) in zip(cols, st.session_state.eval_results.items()):
            with col:
                cm = confusion_matrix(df["true_label"], df["pred_label"], labels=TAGS)
                fig = px.imshow(
                    cm, x=TAGS, y=TAGS, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title=f"{name}", text_auto=True,
                )
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Detailed classification reports"):
            for name, df in st.session_state.eval_results.items():
                st.markdown(f"**{name}**")
                report = classification_report(
                    df["true_label"], df["pred_label"], target_names=TAGS,
                    zero_division=0, output_dict=True,
                )
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        with st.expander("🔍 Per-ticket predictions"):
            for name, df in st.session_state.eval_results.items():
                st.markdown(f"**{name}**")
                st.dataframe(
                    df[["id", "ticket", "true_label", "pred_label", "confidence", "correct"]].style.format(
                        {"confidence": "{:.0%}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

# ──────────────────────────────────────────────────────────────────────────────
# Tab 4 — About
# ──────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.subheader("About this project")
    st.markdown(
        f"""
| Component | Detail |
|-----------|--------|
| **Model** | `{MODEL}` via Groq API (free tier) |
| **Techniques** | Zero-shot · Few-shot · Chain-of-Thought prompting |
| **Tags** | {", ".join(TAGS)} |
| **Priorities** | {", ".join(PRIORITIES)} |
| **Output** | Primary tag + priority + confidence + reasoning |
| **Evaluation** | Accuracy, F1 (macro/weighted), confusion matrix |

### 🧠 Key Insights
1. **Few-shot anchors edge cases** — one labelled example per category improves consistency on ambiguous tickets.
2. **Billing vs Technical overlap** is common (e.g. "can't pay — payment page error"); Chain-of-Thought explicitly reasons through both possibilities.
3. **Confidence scores are meaningful** — clear-cut tickets get higher confidence than ambiguous ones.
4. **Low temperature (0.1)** keeps classification deterministic and reproducible.
5. **Structured JSON output** makes downstream parsing reliable for production routing.

### 🏗️ Suggested production architecture
```
Incoming Ticket
    │
    ▼
Keyword Pre-filter (billing/error/account keywords)
    │
    ├── Clear signal → Few-Shot (fast, cheap)
    └── Ambiguous    → Chain-of-Thought (thorough)
    │
    ▼
Structured JSON Response
    │
    ├── tag        → Route to correct support queue
    ├── priority   → Set SLA timer
    ├── confidence → Flag for human review if < 0.75
    └── reasoning  → Attach to ticket for agent context
```

---
Built from **Task 5 — Auto-Tag Support Tickets**, DevelopersHub Corporation AI/ML Engineering Advanced Internship.
"""
    )
    st.caption("⚠️ Your Groq API key is only used in-session to call the Groq API and is never stored or logged.")
