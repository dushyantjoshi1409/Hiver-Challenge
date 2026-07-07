"""
utils.py — Shared helper utilities for the Hiver AI email challenge.
"""

import os
import json
import re
from dotenv import load_dotenv

load_dotenv()


def get_groq_client():
    """Return a configured Groq client."""
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")
    return Groq(api_key=api_key)


def parse_json_response(text: str) -> dict:
    """
    Robustly parse a JSON object from an LLM response.
    Handles markdown code fences and trailing text gracefully.
    """
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    # Try to extract the first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response:\n{text[:300]}")


def word_count(text: str) -> int:
    """Return the number of words in a string."""
    return len(text.split())


def length_score(reply: str, ideal_min: int = 80, ideal_max: int = 180) -> float:
    """
    Score a reply based on its word count.
    - Full score (1.0) if within [ideal_min, ideal_max].
    - Linearly penalised outside that range, floored at 0.2.
    """
    wc = word_count(reply)
    if ideal_min <= wc <= ideal_max:
        return 1.0
    elif wc < ideal_min:
        # Too short — penalise proportionally
        return max(0.2, wc / ideal_min)
    else:
        # Too long — penalise proportionally
        return max(0.2, ideal_max / wc)


def normalise_score(raw: float, scale: float = 10.0) -> float:
    """Normalise a raw score (out of `scale`) to [0, 1]."""
    return max(0.0, min(1.0, raw / scale))
