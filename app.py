"""
app.py — Streamlit UI for the Hiver AI Email Challenge.

Tabs:
  1. Live Demo     — paste any customer email and get an AI-generated reply + live scores
  2. Batch Results — view the pre-computed results.csv with charts

Usage:
    streamlit run app.py
"""

import os
import time
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from utils import (
    get_groq_client,
    parse_json_response,
    length_score,
    normalise_score,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hiver AI — Email Reply Generator",
    page_icon="✉️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6ee7f7 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .reply-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.65;
        white-space: pre-wrap;
    }
    .overall-badge {
        display: inline-block;
        padding: 0.5rem 1.4rem;
        border-radius: 50px;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        box-shadow: 0 0 24px rgba(99,102,241,0.5);
    }
    .stTextArea textarea {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
    }
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 2rem;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99,102,241,0.45);
    }
</style>
""", unsafe_allow_html=True)

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models…")
def load_resources():
    client = get_groq_client()
    embed  = SentenceTransformer("all-MiniLM-L6-v2")
    return client, embed


MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert customer support agent for a SaaS company.
Your replies must be:
- Empathetic and warm — acknowledge the customer's feelings first.
- Concise — ideally 80–180 words, never more than 250.
- Professional — no slang, no filler phrases like "Certainly!" or "Of course!".
- Action-oriented — always state the next concrete step.
Return ONLY a JSON object with a single key "reply" containing the email reply text."""

JUDGE_SYSTEM = """You are a strict evaluator of customer support email replies.
Return ONLY JSON: {"accuracy": <int 1-10>, "helpfulness": <int 1-10>, "professionalism": <int 1-10>, "empathy": <int 1-10>}"""

TONE_SYSTEM = """Rate the tone of this customer support reply.
Return ONLY JSON: {"tone_score": <int 1-10>}"""


def generate_reply_live(client, email_text: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Customer Email:\n{email_text}"},
        ],
        temperature=0.4, max_tokens=512,
        response_format={"type": "json_object"},
    )
    parsed = parse_json_response(resp.choices[0].message.content)
    return parsed.get("reply", "").strip()


def score_reply_live(client, embed_model, customer_email: str, generated: str) -> dict:
    # Semantic similarity against the generated text itself (self-reference baseline)
    emb = embed_model.encode([customer_email, generated])
    sim = float(np.clip(cosine_similarity([emb[0]], [emb[1]])[0][0], 0, 1))

    # LLM Judge
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": (
                f"Customer email:\n{customer_email}\n\n"
                f"Generated reply:\n{generated}"
            )},
        ],
        temperature=0.0, max_tokens=128,
        response_format={"type": "json_object"},
    )
    j = parse_json_response(resp.choices[0].message.content)
    judge_avg = (j.get("accuracy",5)+j.get("helpfulness",5)+j.get("professionalism",5)+j.get("empathy",5)) / 4.0
    judge_norm = normalise_score(judge_avg, 10.0)

    # Tone
    resp2 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": TONE_SYSTEM},
            {"role": "user", "content": f"Reply:\n{generated}"},
        ],
        temperature=0.0, max_tokens=64,
        response_format={"type": "json_object"},
    )
    t = parse_json_response(resp2.choices[0].message.content)
    tone_norm = normalise_score(float(t.get("tone_score", 5)), 10.0)

    # Length
    length = length_score(generated)

    overall = (0.40*sim + 0.30*judge_norm + 0.15*tone_norm + 0.15*length) * 100

    return {
        "sim":     round(sim, 4),
        "judge":   round(judge_avg, 2),
        "judge_breakdown": j,
        "tone":    round(t.get("tone_score",5), 2),
        "length":  round(length * 10, 2),
        "overall": round(overall, 2),
    }


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">✉️ Hiver AI — Email Reply Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Generates empathetic, professional customer support replies · Evaluates quality with 4 complementary metrics</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🚀 Live Demo", "📊 Batch Results"])

# ── Tab 1: Live Demo ──────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Paste a customer email below")

    sample_emails = {
        "Select a sample…": "",
        "Cancellation request":   "I want to cancel my subscription immediately. I'm not happy with the service at all.",
        "Late delivery":          "I haven't received my order yet. It's been 10 days. This is unacceptable.",
        "App crash":              "The app keeps crashing every time I try to open it on my iPhone.",
        "Double charge":          "I was charged twice for the same order. Please fix this immediately.",
        "Data privacy question":  "I want to know your data privacy policy. Do you sell my data?",
    }
    chosen = st.selectbox("Or pick a sample", list(sample_emails.keys()))

    default_text = sample_emails[chosen]
    customer_email_input = st.text_area(
        "Customer Email",
        value=default_text,
        height=160,
        placeholder="Type the customer's email here…",
        label_visibility="collapsed",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        generate_clicked = st.button("⚡ Generate & Evaluate", use_container_width=True)

    if generate_clicked:
        if not customer_email_input.strip():
            st.warning("Please enter a customer email first.")
        else:
            client, embed_model = load_resources()

            with st.spinner("Generating reply…"):
                try:
                    reply = generate_reply_live(client, customer_email_input)
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.stop()

            with st.spinner("Evaluating quality…"):
                try:
                    scores = score_reply_live(client, embed_model, customer_email_input, reply)
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    st.stop()

            st.markdown("---")
            st.markdown("#### 💬 Generated Reply")
            st.markdown(f'<div class="reply-box">{reply}</div>', unsafe_allow_html=True)

            st.markdown("#### 📈 Quality Scores")
            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                (c1, "Semantic Similarity", f"{scores['sim']:.3f}", "#6ee7f7"),
                (c2, "LLM Judge", f"{scores['judge']} / 10", "#a78bfa"),
                (c3, "Tone Score", f"{scores['tone']} / 10", "#f472b6"),
                (c4, "Length Score", f"{scores['length']} / 10", "#34d399"),
            ]
            for col, label, value, color in metrics:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color:{color}">{value}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            overall_color = "#22c55e" if scores['overall'] >= 80 else "#f59e0b" if scores['overall'] >= 60 else "#ef4444"
            st.markdown(f"""
            <div style="text-align:center; margin-top:0.5rem;">
                <div style="color:#94a3b8; font-size:0.85rem; margin-bottom:0.4rem;">OVERALL SCORE</div>
                <span class="overall-badge" style="background:linear-gradient(135deg,{overall_color}88,{overall_color})">
                    {scores['overall']}%
                </span>
            </div>""", unsafe_allow_html=True)

            with st.expander("🔍 LLM Judge Breakdown"):
                bd = scores["judge_breakdown"]
                bdf = pd.DataFrame([{
                    "Accuracy": bd.get("accuracy","-"),
                    "Helpfulness": bd.get("helpfulness","-"),
                    "Professionalism": bd.get("professionalism","-"),
                    "Empathy": bd.get("empathy","-"),
                }])
                st.dataframe(bdf, use_container_width=True)


# ── Tab 2: Batch Results ──────────────────────────────────────────────────────
with tab2:
    results_path = "results.csv"
    if not os.path.exists(results_path):
        st.info("No `results.csv` found yet. Run `python generate.py` then `python evaluate.py` first.")
    else:
        df = pd.read_csv(results_path)

        avg_overall = df["overall_score"].mean()
        avg_sim     = df["semantic_similarity"].mean()
        avg_judge   = df["llm_judge"].mean()
        avg_tone    = df["tone_score"].mean()

        st.markdown("#### 📊 Aggregate Metrics")
        c1, c2, c3, c4 = st.columns(4)
        agg = [
            (c1, "Avg Similarity",   f"{avg_sim:.3f}",  "#6ee7f7"),
            (c2, "Avg LLM Judge",    f"{avg_judge:.2f}/10", "#a78bfa"),
            (c3, "Avg Tone",         f"{avg_tone:.2f}/10", "#f472b6"),
            (c4, "Overall Accuracy", f"{avg_overall:.1f}%", "#34d399"),
        ]
        for col, label, value, color in agg:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color}">{value}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📋 Per-Email Results")

        display_df = df[["id","overall_score","semantic_similarity","llm_judge","tone_score","length_score"]].copy()
        display_df.columns = ["ID","Overall %","Similarity","LLM Judge /10","Tone /10","Length /10"]
        try:
            import matplotlib  # noqa: F401 — only needed for background_gradient
            styled = display_df.style.background_gradient(subset=["Overall %"], cmap="RdYlGn")
            st.dataframe(styled, use_container_width=True, height=450)
        except ImportError:
            st.dataframe(display_df, use_container_width=True, height=450)

        st.markdown("#### 📈 Score Distribution")
        chart_df = df[["id","overall_score","semantic_similarity"]].set_index("id")
        st.line_chart(chart_df, use_container_width=True)
