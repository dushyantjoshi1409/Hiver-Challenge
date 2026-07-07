"""
evaluate.py — Multi-metric evaluation of generated email replies.

Metrics (weighted composite score):
  40%  Semantic Similarity  — cosine similarity of sentence embeddings
  30%  LLM Judge            — GPT-style rubric: accuracy, helpfulness, professionalism, empathy
  15%  Tone Score           — LLM one-shot tone rating (1–10)
  15%  Length Score         — heuristic based on word count (80–180 words = ideal)

Usage:
    python evaluate.py

Input:
    generated.csv  (output of generate.py)

Output:
    results.csv    (per-email scores + overall accuracy)
"""

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    get_groq_client,
    parse_json_response,
    length_score,
    normalise_score,
)

# ── Configuration ────────────────────────────────────────────────────────────
MODEL          = "llama-3.3-70b-versatile"
EMBED_MODEL    = "all-MiniLM-L6-v2"
INPUT_FILE     = "generated.csv"
OUTPUT_FILE    = "results.csv"

WEIGHTS = {
    "similarity": 0.40,
    "llm_judge":  0.30,
    "tone":       0.15,
    "length":     0.15,
}

SLEEP_BETWEEN_LLM = 0.5   # seconds

# ── Prompt templates ─────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are a strict but fair evaluator of customer support email replies.
Return ONLY a JSON object with exactly these keys and integer values (1–10):
{"accuracy": <int>, "helpfulness": <int>, "professionalism": <int>, "empathy": <int>}"""

JUDGE_USER_TEMPLATE = """Customer email:
{customer_email}

Expected reply (gold standard):
{expected_reply}

Generated reply (to be evaluated):
{generated_reply}

Score the generated reply on each dimension from 1 (very poor) to 10 (excellent):
- accuracy: Does it correctly address the customer's issue?
- helpfulness: Does it provide a concrete next step or resolution?
- professionalism: Is the language professional and error-free?
- empathy: Does it acknowledge the customer's feelings appropriately?"""

TONE_SYSTEM = """You are evaluating the tone of a customer support email.
Return ONLY a JSON object: {"tone_score": <int 1-10>}
10 = exceptionally warm, polite, and professional.
1  = rude, dismissive, or unprofessional."""

TONE_USER_TEMPLATE = """Customer Support Reply:
{reply}

Rate the tone of this reply from 1 to 10."""


# ── Metric functions ──────────────────────────────────────────────────────────

def compute_semantic_similarity(model, expected: str, generated: str) -> float:
    """Cosine similarity between sentence embeddings of expected and generated reply."""
    if not generated.strip():
        return 0.0
    emb = model.encode([expected, generated])
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return float(np.clip(sim, 0.0, 1.0))


def compute_llm_judge(client, customer_email: str, expected: str, generated: str) -> float:
    """
    Ask the LLM to score the reply on 4 rubrics (1–10 each).
    Returns the average normalised to [0, 1].
    """
    if not generated.strip():
        return 0.0
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                    customer_email=customer_email,
                    expected_reply=expected,
                    generated_reply=generated,
                )},
            ],
            temperature=0.0,
            max_tokens=128,
            response_format={"type": "json_object"},
        )
        parsed = parse_json_response(resp.choices[0].message.content)
        scores = [
            parsed.get("accuracy", 5),
            parsed.get("helpfulness", 5),
            parsed.get("professionalism", 5),
            parsed.get("empathy", 5),
        ]
        avg = sum(scores) / len(scores)
        return normalise_score(avg, scale=10.0)
    except Exception as exc:
        print(f"  ⚠ LLM judge error: {exc}")
        return 0.5   # neutral fallback


def compute_tone_score(client, generated: str) -> float:
    """Ask the LLM to rate the tone of the reply (1–10). Returns [0, 1]."""
    if not generated.strip():
        return 0.0
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": TONE_SYSTEM},
                {"role": "user", "content": TONE_USER_TEMPLATE.format(reply=generated)},
            ],
            temperature=0.0,
            max_tokens=64,
            response_format={"type": "json_object"},
        )
        parsed = parse_json_response(resp.choices[0].message.content)
        raw = parsed.get("tone_score", 5)
        return normalise_score(float(raw), scale=10.0)
    except Exception as exc:
        print(f"  ⚠ Tone score error: {exc}")
        return 0.5


def compute_overall(sim: float, judge: float, tone: float, length: float) -> float:
    """Weighted composite score → percentage."""
    score = (
        WEIGHTS["similarity"] * sim
        + WEIGHTS["llm_judge"]  * judge
        + WEIGHTS["tone"]       * tone
        + WEIGHTS["length"]     * length
    )
    return round(score * 100, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")

    # Load embedding model once
    print("Loading sentence-transformer model…")
    embed_model = SentenceTransformer(EMBED_MODEL)

    client = get_groq_client()

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        email_id      = row["id"]
        customer_email = str(row["customer_email"])
        expected      = str(row["expected_reply"])
        generated     = str(row.get("generated_reply", ""))

        # Metric 1 — Semantic Similarity
        sim = compute_semantic_similarity(embed_model, expected, generated)

        # Metric 2 — LLM Judge
        judge = compute_llm_judge(client, customer_email, expected, generated)
        time.sleep(SLEEP_BETWEEN_LLM)

        # Metric 3 — Tone Score
        tone = compute_tone_score(client, generated)
        time.sleep(SLEEP_BETWEEN_LLM)

        # Metric 4 — Length Score
        length = length_score(generated)

        # Composite
        overall = compute_overall(sim, judge, tone, length)

        results.append({
            "id":                  email_id,
            "customer_email":      customer_email,
            "expected_reply":      expected,
            "generated_reply":     generated,
            "semantic_similarity": round(sim, 4),
            "llm_judge":           round(judge * 10, 2),   # display as /10
            "tone_score":          round(tone * 10, 2),    # display as /10
            "length_score":        round(length * 10, 2),  # display as /10
            "overall_score":       overall,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    avg_overall = results_df["overall_score"].mean()
    avg_sim     = results_df["semantic_similarity"].mean()
    avg_judge   = results_df["llm_judge"].mean()
    avg_tone    = results_df["tone_score"].mean()
    avg_len     = results_df["length_score"].mean()

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Semantic Similarity (avg):  {avg_sim:.4f}")
    print(f"  LLM Judge Score (avg /10):  {avg_judge:.2f}")
    print(f"  Tone Score (avg /10):       {avg_tone:.2f}")
    print(f"  Length Score (avg /10):     {avg_len:.2f}")
    print("-" * 55)
    print(f"  ★ Overall Accuracy:         {avg_overall:.2f}%")
    print("=" * 55)
    print(f"\n✅ Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
