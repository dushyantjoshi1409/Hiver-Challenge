"""
generate.py — Generate AI email replies for every row in emails.csv.

Usage:
    python generate.py

Output:
    generated.csv  (columns: id, customer_email, expected_reply, generated_reply)
"""

import json
import time
import pandas as pd
from tqdm import tqdm
from utils import get_groq_client, parse_json_response

# ── Configuration ────────────────────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"       # Groq-hosted Llama 3.3 70B
INPUT_FILE = "emails.csv"
OUTPUT_FILE = "generated.csv"
RETRY_LIMIT = 3
SLEEP_BETWEEN_CALLS = 0.5               # seconds — stay within Groq rate limits

SYSTEM_PROMPT = """You are an expert customer support agent for a SaaS company.
Your replies must be:
- Empathetic and warm — acknowledge the customer's feelings first.
- Concise — ideally 80–180 words, never more than 250.
- Professional — no slang, no filler phrases like "Certainly!" or "Of course!".
- Action-oriented — always state the next concrete step.
- Honest — never make promises you can't keep.

Return ONLY a JSON object with a single key "reply" containing the email reply text.
Do not include any other text outside the JSON."""

USER_PROMPT_TEMPLATE = """Customer Email:
{email}

Write a professional, empathetic customer support reply."""


def generate_reply(client, email_text: str) -> str:
    """Call the LLM and return the generated reply string."""
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(email=email_text)},
                ],
                temperature=0.4,   # low temp → consistent, professional tone
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

    client = get_groq_client()
    generated_replies = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating replies"):
        reply = generate_reply(client, row["customer_email"])
        generated_replies.append(reply)
        time.sleep(SLEEP_BETWEEN_CALLS)

    df["generated_reply"] = generated_replies
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved {len(df)} generated replies to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
