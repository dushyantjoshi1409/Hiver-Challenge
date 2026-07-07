"""
agent.py — Agentic email reply loop.

Loop: Perceive → Act → Observe → Decide → (retry with feedback OR accept)

For each email:
  1. Generate a reply (using few-shot retrieval from dataset)
  2. Evaluate it with 4 metrics → overall %
  3. If score >= 80%: accept and move on
  4. If score < 80%: build specific feedback ("your tone was 6.5/10, too cold")
  5. Retry with that feedback injected into the prompt
  6. Repeat up to MAX_ATTEMPTS times — take the best score found

Usage:
    python agent.py              # batch mode over all 25 emails → agent_results.csv
"""

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_groq_client, length_score
from generate import build_few_shot_block, generate_reply
from evaluate import (
    compute_semantic_similarity,
    compute_llm_judge,
    compute_tone_score,
    compute_overall,
)

# ── Config ────────────────────────────────────────────────────────────────────
SCORE_THRESHOLD = 80.0
MAX_ATTEMPTS    = 3
EMBED_MODEL     = "all-MiniLM-L6-v2"
INPUT_FILE      = "emails.csv"
OUTPUT_FILE     = "agent_results.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_reply(client, embed_model, customer_email, expected, generated) -> dict:
    """Run all 4 metrics. Returns normalised score dict."""
    sim    = compute_semantic_similarity(embed_model, expected, generated)
    judge  = compute_llm_judge(client, customer_email, expected, generated)
    time.sleep(0.3)
    tone   = compute_tone_score(client, generated)
    time.sleep(0.3)
    length = length_score(generated)
    return {
        "similarity": round(sim, 4),
        "judge":      round(judge * 10, 2),
        "tone":       round(tone  * 10, 2),
        "length":     round(length * 10, 2),
        "overall":    round(compute_overall(sim, judge, tone, length), 2),
    }


def build_feedback(scores: dict, attempt_num: int) -> str:
    """Turn low scores into specific, actionable feedback for the next attempt."""
    issues = []
    if scores["similarity"] < 0.75:
        issues.append("semantic alignment is low — address the customer's specific concern directly, avoid being generic")
    if scores["judge"] < 8.0:
        issues.append(f"quality rated {scores['judge']:.1f}/10 — be more accurate, helpful, and empathetic")
    if scores["tone"] < 8.0:
        issues.append(f"tone rated {scores['tone']:.1f}/10 — use a warmer, more reassuring professional tone")
    if scores["length"] < 8.0:
        issues.append("reply length was off — aim for exactly 80–180 words")
    if not issues:
        issues.append("overall quality needs improvement — be more precise and action-oriented")

    lines = [f"Attempt {attempt_num} scored {scores['overall']:.1f}% — below the 80% threshold. Fix these issues:"]
    for i, issue in enumerate(issues, 1):
        lines.append(f"  {i}. {issue}")
    lines.append("\nWrite a significantly improved reply addressing every point above.")
    return "\n".join(lines)


# ── Core agent function ───────────────────────────────────────────────────────

def run_agent_for_email(
    client,
    embed_model,
    dataset_df: pd.DataFrame,
    dataset_embeddings,
    customer_email: str,
    expected_reply: str,
    row_idx: int = 0,
    threshold: float = SCORE_THRESHOLD,
    max_attempts: int = MAX_ATTEMPTS,
    progress_callback=None,          # fn(attempt_num, reply, scores, accepted, feedback)
) -> dict:
    """
    Agentic loop for ONE email.

    progress_callback is called after every attempt — used by Streamlit
    to display live step-by-step agent activity.

    Returns:
        best_reply, best_score, total_attempts, accepted_at_attempt, attempts[]
    """
    attempts      = []
    feedback_text = None

    for attempt_num in range(1, max_attempts + 1):

        # ── PERCEIVE: retrieve top-3 similar past emails ──────────────────────
        q_emb = embed_model.encode([customer_email])
        sims  = cosine_similarity(q_emb, dataset_embeddings)[0]
        sims[row_idx] = -1.0                              # exclude self
        top_k     = np.argsort(sims)[::-1][:3]
        few_shot  = build_few_shot_block(dataset_df.iloc[top_k])

        # ── ACT: generate reply (with feedback injected if retry) ─────────────
        context = few_shot
        if feedback_text:
            context += f"\n\n--- Self-Correction Feedback ---\n{feedback_text}\n"
        reply = generate_reply(client, context, customer_email)
        time.sleep(0.5)

        # ── OBSERVE: evaluate ─────────────────────────────────────────────────
        scores   = score_reply(client, embed_model, customer_email, expected_reply, reply)
        accepted = scores["overall"] >= threshold

        record = {
            "attempt":        attempt_num,
            "reply":          reply,
            "scores":         scores,
            "feedback_given": feedback_text,
            "accepted":       accepted,
        }
        attempts.append(record)

        if progress_callback:
            progress_callback(attempt_num, reply, scores, accepted, feedback_text)

        # ── DECIDE: accept or build feedback for next attempt ─────────────────
        if accepted:
            break
        if attempt_num < max_attempts:
            feedback_text = build_feedback(scores, attempt_num)

    best = max(attempts, key=lambda a: a["scores"]["overall"])
    return {
        "best_reply":          best["reply"],
        "best_score":          best["scores"]["overall"],
        "accepted_at_attempt": best["attempt"],
        "total_attempts":      len(attempts),
        "attempts":            attempts,
    }


# ── Batch main ────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} emails")

    embed_model        = SentenceTransformer(EMBED_MODEL)
    dataset_embeddings = embed_model.encode(df["customer_email"].tolist(), show_progress_bar=True)
    client             = get_groq_client()
    records            = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Agent"):
        print(f"\n  Email #{row['id']}: {row['customer_email'][:55]}…")
        result = run_agent_for_email(
            client, embed_model, df, dataset_embeddings,
            row["customer_email"], row["expected_reply"], row_idx=idx,
        )
        print(f"  → {result['total_attempts']} attempt(s) | Score: {result['best_score']:.2f}%")
        records.append({
            "id":                  row["id"],
            "customer_email":      row["customer_email"],
            "expected_reply":      row["expected_reply"],
            "best_reply":          result["best_reply"],
            "best_score":          result["best_score"],
            "total_attempts":      result["total_attempts"],
            "accepted_at_attempt": result["accepted_at_attempt"],
        })

    out = pd.DataFrame(records)
    out.to_csv(OUTPUT_FILE, index=False)
    avg   = out["best_score"].mean()
    retry = (out["total_attempts"] > 1).sum()
    print(f"\n{'='*50}")
    print(f"  ★ Agent Overall Accuracy: {avg:.2f}%")
    print(f"  Emails that needed retry: {retry} / {len(df)}")
    print(f"{'='*50}")
    print(f"✅ Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
