"""
generate.py — Generate AI email replies grounded in the dataset.

Approach: Semantic Few-Shot Retrieval
  For each incoming email, we retrieve the top-K most semantically similar
  past emails from emails.csv using sentence embeddings + cosine similarity.
  Those retrieved pairs are injected into the prompt as few-shot examples,
  grounding the generation in real historical response patterns.

  This is a lightweight form of RAG (Retrieval-Augmented Generation) that
  avoids the overhead of a vector database while still anchoring the model
  to the style, tone, and structure of actual past responses.

Trade-offs vs alternatives:
  - Few-shot retrieval (chosen): fast, no infra, transparent, works well
    with small datasets. Slightly increases prompt token usage.
  - Full RAG with vector DB: better at scale (1000s of emails), overkill here.
  - Fine-tuning: highest quality ceiling but requires GPU, data labelling,
    and retraining cycles — not appropriate for a 25-example dataset.

Usage:
    python generate.py

Output:
    generated.csv  (columns: id, customer_email, expected_reply, generated_reply)
"""

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_groq_client, parse_json_response

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL        = "llama-3.3-70b-versatile"
EMBED_MODEL  = "all-MiniLM-L6-v2"
INPUT_FILE   = "emails.csv"
OUTPUT_FILE  = "generated.csv"
TOP_K        = 3          # number of few-shot examples to retrieve per email
RETRY_LIMIT  = 3
SLEEP_BETWEEN_CALLS = 0.5

SYSTEM_PROMPT = """You are an expert customer support agent for a SaaS company.
You will be given a few examples of real past customer emails and the replies that were sent.
Use those examples to learn the correct tone, structure, and level of detail — then write a reply for the new email.

Your replies must be:
- Empathetic and warm — acknowledge the customer's feelings first.
- Concise — ideally 80–180 words, never more than 250.
- Professional — no slang, no filler phrases like "Certainly!" or "Of course!".
- Action-oriented — always state the next concrete step.
- Honest — never make promises you can't keep.

Return ONLY a JSON object with a single key "reply" containing your email reply."""


def build_few_shot_block(similar_rows: pd.DataFrame) -> str:
    """Format retrieved similar pairs into a few-shot prompt block."""
    lines = ["Here are examples of past customer emails and the replies that were sent:\n"]
    for i, (_, row) in enumerate(similar_rows.iterrows(), 1):
        lines.append(f"--- Example {i} ---")
        lines.append(f"Customer Email: {row['customer_email']}")
        lines.append(f"Reply Sent: {row['expected_reply']}\n")
    return "\n".join(lines)


def generate_reply(client, few_shot_block: str, email_text: str) -> str:
    """Call the LLM with few-shot context and return the generated reply."""
    user_content = (
        f"{few_shot_block}\n"
        f"--- New Email to Reply To ---\n"
        f"Customer Email: {email_text}\n\n"
        f"Write a reply following the same tone and style as the examples above."
    )

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.4,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            parsed = parse_json_response(raw)
            return parsed.get("reply", "").strip()

        except Exception as exc:
            if attempt == RETRY_LIMIT:
                print(f"  ✗ Failed after {RETRY_LIMIT} attempts: {exc}")
                return ""
            wait = 2 ** attempt
            print(f"  ⚠ Attempt {attempt} failed ({exc}). Retrying in {wait}s…")
            time.sleep(wait)


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} emails from {INPUT_FILE}")

    # Pre-compute embeddings for all emails in the dataset
    print("Computing dataset embeddings for few-shot retrieval…")
    embed_model = SentenceTransformer(EMBED_MODEL)
    dataset_embeddings = embed_model.encode(df["customer_email"].tolist(), show_progress_bar=True)

    client = get_groq_client()
    generated_replies = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating replies"):
        # Embed the current email
        query_emb = embed_model.encode([row["customer_email"]])

        # Compute cosine similarity to ALL other emails in the dataset
        sims = cosine_similarity(query_emb, dataset_embeddings)[0]

        # Exclude the email itself (it would be a perfect match)
        sims[idx] = -1.0

        # Retrieve top-K most similar examples
        top_k_indices = np.argsort(sims)[::-1][:TOP_K]
        similar_rows = df.iloc[top_k_indices]

        # Build the few-shot block and generate
        few_shot_block = build_few_shot_block(similar_rows)
        reply = generate_reply(client, few_shot_block, row["customer_email"])
        generated_replies.append(reply)

        time.sleep(SLEEP_BETWEEN_CALLS)

    df["generated_reply"] = generated_replies
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved {len(df)} generated replies to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
