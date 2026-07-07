# ✉️ Hiver AI — Email Reply Generator & Evaluator

> **An end-to-end AI pipeline that generates professional customer support email replies and evaluates them using a multi-metric scoring system — fully runnable in three commands.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model: Llama 3.3 70B](https://img.shields.io/badge/model-Llama%203.3%2070B-blueviolet)](https://groq.com)
[![Streamlit UI](https://img.shields.io/badge/UI-Streamlit-ff4b4b)](https://streamlit.io)
[![Overall Accuracy](https://img.shields.io/badge/Overall%20Accuracy-80.40%25-brightgreen)](./results.csv)

---

## Table of Contents

- [What This Does](#what-this-does)
- [Architecture](#architecture)
- [Prompting Strategy](#prompting-strategy)
- [Evaluation Methodology](#evaluation-methodology)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Actual Results](#actual-results)
- [Project Structure](#project-structure)
- [Tools & Libraries](#tools--libraries)
- [Design Decisions](#design-decisions)
- [Future Improvements](#future-improvements)

---

## What This Does

This project solves two distinct problems that directly mirror what Hiver does at scale:

| Problem | Solution |
|---------|----------|
| **Dataset** | 25 hand-authored synthetic email/reply pairs covering the most common customer support scenarios for a SaaS company |
| **Response Generation** | Given a new email, semantically retrieve the top-3 most similar past emails from the dataset, inject them as few-shot examples, then call Llama 3.3 70B to generate a grounded reply |
| **Response Evaluation** | Each generated reply is scored across four complementary metrics and combined into a single weighted percentage score |

Both the generator and evaluator are built as standalone scripts (`generate.py`, `evaluate.py`) that are fully runnable end-to-end. A **Streamlit UI** (`app.py`) lets you generate and evaluate replies live in a browser.

---

## Dataset

### Source & Construction

The dataset (`emails.csv`) contains **25 synthetic customer support email/reply pairs**, hand-authored to be representative of real SaaS customer interactions.

**Why synthetic?** Real customer support emails contain PII (names, order numbers, account details) and are proprietary. Synthetic data lets us control coverage, avoid privacy issues, and ensure the dataset is shareable in a public repository.

**Why representative?** The 25 scenarios were chosen by mapping to the most frequent customer support request categories in SaaS:

| Category | Examples in dataset |
|----------|--------------------|
| Billing & Payments | Double charge, wrong invoice, refund, tax invoice |
| Subscription Management | Cancel, upgrade, downgrade, loyalty discount |
| Delivery & Orders | Late order, wrong address, damaged item, missing package |
| Account & Access | Password reset, deleted account, multi-user setup |
| Technical Issues | App crash, API integration, slow website |
| Product & Policy | Data privacy, free trial, data export |

Each reply was written to reflect real support team standards: empathetic opening, clear action, professional tone, 80–180 word target length.

**How AI tools were used in dataset creation:** Scenario topics were brainstormed with ChatGPT, but all 25 email/reply pairs were written manually and reviewed for quality, coverage, and realism before being committed.

---

## Architecture

```
emails.csv  (25 labelled email/reply pairs)
      │
      ▼
 generate.py
      │
      ├── 1. Embed all 25 emails with all-MiniLM-L6-v2
      │
      ├── 2. For each target email:
      │        → compute cosine similarity to all other emails
      │        → retrieve top-3 most similar past pairs (excluding self)
      │        → build few-shot prompt block with those 3 examples
      │
      └── 3. Call Groq / Llama 3.3 70B with:
               system prompt + few-shot examples + target email
               → JSON response parsed → stored per row
      ▼
 generated.csv  (customer_email | expected_reply | generated_reply)
      │
      ▼
 evaluate.py
      ├── Metric 1: Semantic Similarity   [weight: 40%]
      │     sentence-transformers · all-MiniLM-L6-v2 · cosine similarity
      │
      ├── Metric 2: LLM Judge             [weight: 30%]
      │     Groq / Llama 3.3 70B · 4-rubric JSON rubric (1–10 each)
      │     accuracy + helpfulness + professionalism + empathy
      │
      ├── Metric 3: Tone Score            [weight: 15%]
      │     Groq / Llama 3.3 70B · focused single-prompt tone rating (1–10)
      │
      └── Metric 4: Length Score          [weight: 15%]
            heuristic · 80–180 words = 10/10 · linear penalty outside range
      │
      ▼
 results.csv  ──→  app.py  ──→  Streamlit UI
                                  ├── Live Demo tab (generate + score any email)
                                  └── Batch Results tab (charts + table)
```

---

## Prompting Strategy

### Generation Approach: Semantic Few-Shot Retrieval

The generator does **not** use a generic standalone prompt. Instead, it grounds every reply in the dataset through **semantic few-shot retrieval**:

1. All 25 emails in the dataset are embedded with `all-MiniLM-L6-v2`
2. For each incoming email, cosine similarity is computed against all other emails in the dataset
3. The top-3 most similar past email/reply pairs are retrieved
4. Those 3 pairs are injected into the prompt as concrete few-shot examples
5. The LLM generates a reply that mirrors the style, tone, and structure of the retrieved examples

**Why this approach over alternatives:**

| Approach | Chosen? | Reason |
|----------|---------|--------|
| **Semantic Few-Shot Retrieval** | ✅ Yes | Grounds generation in dataset. Fast, no infra needed, transparent. Works well at 25-example scale. |
| Full RAG with vector DB (Pinecone, Chroma) | ❌ No | Overkill for 25 examples. Adds infrastructure complexity with no quality gain at this scale. |
| Fine-tuning | ❌ No | Requires GPU, data labelling pipeline, and retraining cycles. 25 examples is far too few — would overfit badly. |
| Static few-shot (same examples every time) | ❌ No | Less relevant examples hurt quality. Semantic retrieval ensures the most applicable examples are always chosen. |

### System Prompt

```
You are an expert customer support agent for a SaaS company.
You will be given a few examples of real past customer emails and the replies
that were sent. Use those examples to learn the correct tone, structure,
and level of detail — then write a reply for the new email.

Your replies must be:
- Empathetic and warm — acknowledge the customer's feelings first.
- Concise — ideally 80–180 words, never more than 250.
- Professional — no slang, no filler phrases like "Certainly!" or "Of course!".
- Action-oriented — always state the next concrete step.
- Honest — never make promises you can't keep.

Return ONLY a JSON object with a single key "reply" containing your email reply.
```

Key prompt decisions:
- **JSON output mode** (`response_format: json_object`) makes parsing deterministic, no regex hacks.
- **Temperature 0.4** — consistent and professional, but not so low it produces repetitive boilerplate.
- **Anti-patterns called out explicitly** ("Certainly!", "Of course!") — these filler openers are common in support models and reduce quality perception.
- **Explicit few-shot framing** ("use these examples to learn tone and structure") steers the model to treat the retrieved pairs as style guidance, not just information.

### LLM Judge Prompt

The judge uses a separate LLM call with a zero-temperature, structured JSON rubric:

```
You are evaluating an email reply.
Score the generated reply from 1–10 on:
  - accuracy:        Does it correctly address the customer's issue?
  - helpfulness:     Does it provide a concrete next step or resolution?
  - professionalism: Is the language professional and error-free?
  - empathy:         Does it acknowledge the customer's feelings?

Return ONLY JSON: {"accuracy": int, "helpfulness": int, "professionalism": int, "empathy": int}
```

---

## Evaluation Methodology

### Why multiple metrics?

String-matching (BLEU, exact match) is insufficient for open-ended generative tasks:
- A reply can be **semantically correct** using completely different words.
- A reply can have **high lexical overlap** with the reference while being tone-deaf or unhelpfully terse.
- An LLM judge alone introduces **model bias and variance** — it shouldn't be the sole signal.

We combine four **orthogonal, complementary** signals:

| Metric | Weight | Rationale |
|--------|--------|-----------|
| **Semantic Similarity** | **40%** | Cosine similarity of `all-MiniLM-L6-v2` sentence embeddings between expected and generated reply. Captures *meaning*, not wording. Objective and reproducible. Anchors the score. |
| **LLM Judge** | **30%** | Four-rubric scoring: accuracy, helpfulness, professionalism, empathy. Captures qualitative nuances that similarity misses (e.g. "did it actually tell the customer what to do next?"). Weighted below similarity because LLM scores carry stochastic variance. |
| **Tone Score** | **15%** | A focused, isolated prompt rates tone warmth and professionalism (1–10). Intentionally separate from the judge so tone quality doesn't get diluted by the 4-rubric average. |
| **Length Score** | **15%** | Heuristic: 80–180 words = 10/10. Linearly penalised below 80 (too terse) or above 180 (too verbose). Guards against both dismissive one-liners and meandering essays. Floor at 2/10. |

**Final Score = 40% · Similarity + 30% · Judge + 15% · Tone + 15% · Length → as a percentage**

### Why these weights?

- **Semantic similarity (40%)** is the most objective, reproducible signal. It acts as the anchor.
- **LLM judge (30%)** provides rich qualitative signal but is subject to inter-run variance, so it's influential but not dominant.
- **Tone (15%)** and **Length (15%)** are important hygiene checks that prevent obviously bad replies from scoring well. They're capped at 15% each so they can't override a semantically strong reply.

### Why not just BLEU or ROUGE?

BLEU/ROUGE measure n-gram overlap. In customer support email, two excellent replies to the same query can share almost zero word overlap. For example, "We'll refund you within 3 days" and "A refund will be processed to your account in 72 hours" are semantically identical but have near-zero BLEU score. Sentence embeddings handle this correctly.

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/dushyantjoshi1409/Hiver-Challenge.git
cd Hiver-Challenge
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root (never committed to git):

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a **free** Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

---

## How to Run

### Step 1 — Generate AI replies (~45 seconds for 25 emails)

```bash
python generate.py
```

**What it does:** Reads `emails.csv` → calls Groq/Llama 3.3 70B for each row → saves `generated.csv`

### Step 2 — Evaluate the replies (~2 minutes for 25 emails)

```bash
python evaluate.py
```

**What it does:** Reads `generated.csv` → computes all 4 metrics → saves `results.csv` → prints summary to terminal:

```
=======================================================
  EVALUATION RESULTS
=======================================================
  Semantic Similarity (avg):  0.7253
  LLM Judge Score (avg /10):  8.39
  Tone Score (avg /10):       9.20
  Length Score (avg /10):     8.28
-------------------------------------------------------
  ★ Overall Accuracy:         80.40%
=======================================================
```

### Step 3 (Bonus) — Launch the Streamlit UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with two tabs:

| Tab | What you can do |
|-----|----------------|
| **🚀 Live Demo** | Paste any customer email → click Generate & Evaluate → see the AI reply + all 4 scores in real time |
| **📊 Batch Results** | View the pre-computed `results.csv` with aggregate metrics and a per-email score chart |

> **Note:** `generated.csv` and `results.csv` are already committed so the Batch Results tab works immediately without running Steps 1 & 2.

---

## Actual Results

These are the real numbers produced by running the full pipeline on the 25-email dataset:

| ID | Customer Issue (short) | Similarity | LLM Judge /10 | Tone /10 | Length /10 | **Overall %** |
|----|------------------------|:----------:|:-------------:|:--------:|:----------:|:-------------:|
| 1 | Cancel subscription | 0.7785 | 8.50 | 9.0 | 8.0 | 82.14 |
| 2 | Late order (10 days) | 0.7034 | 8.00 | 9.0 | 6.75 | 75.76 |
| 3 | Change shipping address | 0.6620 | 9.00 | 9.0 | 7.12 | 77.67 |
| 4 | Double charge | 0.7959 | 7.75 | 10.0 | 7.62 | 81.52 |
| 5 | Product broke in 2 weeks | 0.7370 | 8.75 | 10.0 | 7.75 | 82.35 |
| 6 | Refund – wrong product | 0.6950 | 8.75 | 9.0 | 7.38 | 78.61 |
| 7 | Password reset | 0.7868 | 9.00 | 9.0 | 9.12 | 85.66 |
| 8 | Plan upgrade options | 0.7144 | 7.50 | 9.0 | 8.38 | 77.14 |
| 9 | Loyalty discount | 0.6606 | 8.00 | 9.0 | 8.0 | 75.93 |
| 10 | App crashing on iPhone | 0.9314 | 8.75 | 9.0 | 8.88 | **90.32** |
| 11 | Recover deleted account | 0.8324 | 9.25 | 10.0 | 7.62 | 87.48 |
| 12 | Free trial inquiry | 0.6841 | 8.75 | 9.0 | 9.62 | 81.55 |
| 13 | API integration help | 0.6241 | 7.75 | 9.0 | 8.0 | 73.72 |
| 14 | Wrong invoice amount | 0.8737 | 8.50 | 9.0 | 9.88 | 88.76 |
| 15 | Plan downgrade | 0.6525 | 7.50 | 9.0 | 7.38 | 73.16 |
| 16 | Package missing (1 week) | 0.6937 | 7.75 | 9.0 | 7.62 | 75.93 |
| 17 | Damaged item | 0.6648 | 8.25 | 9.0 | 7.75 | 76.47 |
| 18 | Team / multi-user access | 0.7908 | 8.25 | 9.0 | 10.0 | 84.88 |
| 19 | Missed emails (unsubscribed) | 0.6452 | 7.75 | 10.0 | 8.25 | 76.43 |
| 20 | Slow website / checkout | 0.6796 | 9.25 | 9.0 | 9.0 | 81.93 |
| 21 | Data export | 0.8338 | 8.75 | 9.0 | 9.0 | 86.60 |
| 22 | Data privacy / selling data | 0.7286 | 8.50 | 10.0 | 9.38 | 83.71 |
| 23 | Payment failed but charged | 0.7591 | 8.75 | 9.0 | 8.88 | 83.43 |
| 24 | Tax invoices for 3 purchases | 0.7032 | 7.75 | 9.0 | 7.50 | 76.13 |
| 25 | 1 hour on hold – bad support | 0.5008 | 9.00 | 9.0 | 8.12 | 72.72 |
| **Average** | | **0.7253** | **8.39** | **9.20** | **8.28** | **80.40%** |

### Observations

- **Tone scores are consistently high (9.2 avg)** — the system prompt's explicit guidance ("acknowledge feelings first", "no filler phrases") is effective.
- **Semantic similarity is the most variable metric** (range: 0.50–0.93) — expected, since the generator is free to paraphrase.
- **Length is the main drag on score** — the model sometimes asks for additional info instead of acting directly, producing slightly shorter replies than ideal.
- **Lowest score (72.72%) on "1 hour on hold"** — the generated reply was empathetic but didn't match the reference's urgency and commitment level, reducing similarity.

---

## Project Structure

```
Hiver-Challenge/
├── README.md           ← This file
├── requirements.txt    ← All Python dependencies
├── .env                ← Your API key (not committed — add this yourself)
├── .gitignore          ← Excludes .env, venv/, __pycache__/
│
├── emails.csv          ← Dataset: 25 customer support email/reply pairs
├── generate.py         ← Step 1: LLM reply generator
├── evaluate.py         ← Step 2: 4-metric evaluation pipeline
├── utils.py            ← Shared helpers: Groq client, JSON parsing, scoring
├── app.py              ← Streamlit UI: live demo + batch results
│
├── generated.csv       ← Pre-generated AI replies (output of generate.py)
└── results.csv         ← Pre-computed evaluation scores (output of evaluate.py)
```

---

## Tools & Libraries

| Tool | Version | Purpose |
|------|---------|---------|
| [Groq](https://groq.com) | latest | LLM inference — Llama 3.3 70B for generation, judging, and tone scoring |
| [sentence-transformers](https://www.sbert.net) | latest | `all-MiniLM-L6-v2` embeddings for semantic similarity |
| [scikit-learn](https://scikit-learn.org) | latest | `cosine_similarity` computation |
| [pandas](https://pandas.pydata.org) | latest | CSV I/O and data manipulation |
| [numpy](https://numpy.org) | latest | Numerical operations |
| [streamlit](https://streamlit.io) | latest | Interactive web UI |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | latest | Secure API key management via `.env` |
| [tqdm](https://tqdm.github.io) | latest | Progress bars during batch processing |
| [matplotlib](https://matplotlib.org) | latest | Required by pandas Styler for heatmaps |

---

## Design Decisions

**Why Groq + Llama 3.3 70B instead of OpenAI?**
Groq offers a generous free tier with very fast inference (~200 tokens/sec). Llama 3.3 70B performs comparably to GPT-4o on instruction-following benchmarks. Using one provider for both generation and evaluation keeps the stack simple and dependency-free from paid credits.

**Why `all-MiniLM-L6-v2` for embeddings?**
It's a lightweight (22M parameter) model that runs entirely locally — no API call needed. Despite its small size, it achieves strong results on semantic similarity benchmarks (SBERT leaderboard) and keeps the evaluation pipeline fast.

**Why 25 examples?**
Sufficient to observe patterns (high-tone consistency, length variance, similarity spread) without requiring hours of evaluation API time. The pipeline scales linearly — running on 500 examples requires no code changes, just a larger CSV.

**Why is the `.env` file not committed?**
Security best practice. Anyone cloning this repo creates their own `.env` with their own Groq key. The `.gitignore` explicitly excludes `.env`.

---

## How AI Tools Were Used

The challenge asks for transparency on AI tool usage. Here's an honest account:

| Task | Tool Used | How |
|------|-----------|-----|
| Scenario brainstorming | ChatGPT | Generated a list of the most common SaaS customer support request types to ensure dataset coverage |
| Email/reply writing | Human (manual) | All 25 email/reply pairs were written and reviewed by hand. AI suggestions were not used for the actual content |
| LLM inference (generation) | Groq / Llama 3.3 70B | Core AI for generating email replies at runtime |
| LLM inference (evaluation) | Groq / Llama 3.3 70B | LLM judge and tone scorer in the evaluation pipeline |
| Sentence embeddings | all-MiniLM-L6-v2 (local) | Semantic similarity for retrieval and evaluation — runs fully offline |
| Code assistance | Antigravity (AI coding assistant) | Used for boilerplate, debugging, and code structure — all logic, prompts, and evaluation design are the author's own |

**Bottom line:** The evaluation methodology, prompting strategy, metric weights, and architectural decisions are original work. AI tools were used as productivity accelerators, not as a substitute for thinking.

---

## Future Improvements

- **Reference-free evaluation** — Replace the expected-reply comparison with a trained reward model (e.g., `OpenAssistant/reward-model-deberta-v3-large-v2`) to evaluate without gold labels.
- **A/B prompt testing** — Run multiple system prompt variants through the same evaluation pipeline to empirically identify the best phrasing.
- **Hallucination detection** — Add an NLI-based factual consistency check to flag replies that fabricate order numbers, timelines, or policies.
- **Async evaluation** — Parallelise the LLM judge and tone calls with `asyncio` + `httpx` to cut evaluation time from ~2 minutes to ~30 seconds.
- **BLEU / ROUGE as baseline** — Include n-gram metrics as a cheap sanity check alongside neural metrics, making the composite score more interpretable.
- **Domain fine-tuning** — Fine-tune a smaller model (e.g., Mistral 7B) on company-specific tone and policy data for production deployment.
- **Confidence intervals** — Run each LLM judge call 3× and report mean ± std to quantify evaluation variance.
