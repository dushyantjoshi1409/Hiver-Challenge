# Hiver AI — Email Reply Generator & Evaluator

> **An end-to-end AI pipeline that generates professional customer support email replies and evaluates them with a multi-metric scoring system.**

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Evaluation Methodology](#evaluation-methodology)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Sample Results](#sample-results)
- [Tools & Libraries](#tools--libraries)
- [Future Improvements](#future-improvements)

---

## Overview

This project solves two distinct problems:

1. **Response Generation** — Given a raw customer email, an LLM generates an empathetic, professional, concise reply.
2. **Response Evaluation** — Each reply is scored on four complementary metrics and combined into a single percentage score.

The pipeline is fully runnable end-to-end via three commands and also ships with a **Streamlit UI** for live interactive demos.

---

## Architecture

```
emails.csv
    │
    ▼
generate.py  ──→  Groq / Llama-3.3-70B  ──→  generated.csv
    │
    ▼
evaluate.py
    ├── Metric 1: Semantic Similarity  (sentence-transformers / all-MiniLM-L6-v2)
    ├── Metric 2: LLM Judge            (Groq / Llama-3.3-70B · 4-rubric JSON)
    ├── Metric 3: Tone Score           (Groq / Llama-3.3-70B · 1-10 rating)
    └── Metric 4: Length Score         (heuristic · 80–180 words = ideal)
    │
    ▼
results.csv  ──→  (optional) app.py  ──→  Streamlit UI
```

---

## Evaluation Methodology

### Why multiple metrics?

String-matching alone is insufficient for open-ended text generation. A reply can be **semantically correct** while using completely different words. Conversely, a reply can have **high word overlap** with the reference while being tone-deaf or unhelpfully long.

We combine four orthogonal signals:

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| **Semantic Similarity** | 40% | Cosine similarity of sentence embeddings (expected vs. generated). Captures meaning, not just wording. |
| **LLM Judge** | 30% | A second LLM pass scores the reply on Accuracy, Helpfulness, Professionalism, and Empathy (1–10 each). Captures nuanced quality a similarity score misses. |
| **Tone Score** | 15% | A focused prompt asks the LLM to rate the warmth and professionalism of the tone alone (1–10). |
| **Length Score** | 15% | Heuristic: 80–180 words = full marks, linearly penalised outside that range. Guards against one-line dismissals and 500-word essays. |

**Final score = 40%·Similarity + 30%·Judge + 15%·Tone + 15%·Length** → expressed as a percentage.

### Why these weights?

- Semantic similarity is the most objective signal and acts as the anchor (40%).
- The LLM judge provides the richest qualitative feedback but is subject to LLM variance, so it gets a 30% weight rather than being the sole arbiter.
- Tone and Length are important hygiene checks but shouldn't dominate — 15% each keeps them influential without over-indexing.

### Why Groq + Llama 3.3?

- Zero cost during prototyping with generous free-tier rate limits.
- Llama 3.3 70B performs comparably to GPT-4o on instruction-following tasks.
- The same model is used for generation and evaluation, keeping the stack simple.

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/hiver-ai.git
cd hiver-ai
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

### 4. Configure environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at https://console.groq.com

---

## How to Run

### Step 1 — Generate AI replies

```bash
python generate.py
```

Reads `emails.csv` → calls Groq API → writes `generated.csv`.

### Step 2 — Evaluate the replies

```bash
python evaluate.py
```

Reads `generated.csv` → computes 4 metrics → writes `results.csv` → prints overall accuracy.

### Step 3 (Optional) — Launch the Streamlit UI

```bash
streamlit run app.py
```

Opens a browser with:
- **Live Demo tab** — paste any customer email, generate a reply, and see scores in real time.
- **Batch Results tab** — view the pre-computed `results.csv` with charts.

---

## Sample Results

| ID | Semantic Similarity | LLM Judge /10 | Tone /10 | Length /10 | Overall % |
|----|--------------------:|:-------------:|:--------:|:----------:|:---------:|
| 1  | 0.8812 | 9.2 | 9.5 | 9.4 | 91.2 |
| 2  | 0.8634 | 8.8 | 9.0 | 10  | 89.4 |
| 3  | 0.9231 | 9.5 | 10  | 9.2 | 94.1 |
| …  | …     | …  | …  | …  | …   |
| **Avg** | **0.887** | **9.1** | **9.4** | **9.6** | **91.8%** |

> Actual numbers will vary depending on the LLM's stochastic output.

---

## Project Structure

```
hiver-ai/
├── README.md           ← This file
├── requirements.txt    ← Python dependencies
├── .env                ← API keys (not committed)
├── .gitignore
├── emails.csv          ← 25 hand-crafted email/reply pairs (dataset)
├── generate.py         ← Step 1: generate AI replies
├── evaluate.py         ← Step 2: evaluate with 4 metrics
├── utils.py            ← Shared helpers (Groq client, JSON parsing, scoring)
├── app.py              ← Step 3: Streamlit UI
├── generated.csv       ← Output of generate.py
└── results.csv         ← Output of evaluate.py
```

---

## Tools & Libraries

| Tool | Purpose |
|------|---------|
| [Groq](https://groq.com) | LLM inference (Llama 3.3 70B) — used for generation, judging, and tone scoring |
| [sentence-transformers](https://www.sbert.net) | `all-MiniLM-L6-v2` embeddings for semantic similarity |
| [scikit-learn](https://scikit-learn.org) | `cosine_similarity` computation |
| [pandas](https://pandas.pydata.org) | CSV I/O and data manipulation |
| [numpy](https://numpy.org) | Numerical operations |
| [streamlit](https://streamlit.io) | Interactive web UI |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Environment variable management |
| [tqdm](https://tqdm.github.io) | Progress bars |

---

## Future Improvements

- **Reference-free evaluation** — Use a trained reward model instead of an LLM judge to reduce bias and cost.
- **A/B prompt testing** — Compare multiple system prompts side-by-side with the evaluation pipeline.
- **Hallucination detection** — Add a factual consistency check (e.g., using NLI models) to flag replies that fabricate information.
- **Domain-specific fine-tuning** — Fine-tune a smaller model on company-specific tone and policy data.
- **BLEU / ROUGE baseline** — Add classic n-gram metrics as a low-cost sanity check alongside neural metrics.
- **Async evaluation** — Parallelise LLM judge and tone calls to cut evaluation time by ~50%.
