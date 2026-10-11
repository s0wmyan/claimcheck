"""Ask a local language model whether a claim is supported by abstracts."""
import json

import requests

from claimcheck.data import LABELS
from claimcheck.dense import OLLAMA_URL

CHAT_MODEL = "phi4-mini"
PARSE_FAILED = "Could not read the model's reply."

# Ollama forces the reply to match this shape. The reason comes first so the
# model works through the evidence before committing to a verdict.
REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": LABELS},
    },
    "required": ["reason", "verdict"],
}

OPEN_BOOK = """You are checking a scientific claim against paper abstracts.

Claim: {claim}

{abstracts}

Using only these abstracts, choose one verdict:
- SUPPORTED: an abstract gives evidence that the claim is true.
- REFUTED: an abstract gives evidence that the claim is false.
- NOT_ENOUGH_INFO: the abstracts do not settle the claim either way.

Give a one-sentence reason, then the verdict."""

CLOSED_BOOK = """You are checking a scientific claim using only what you already know.

Claim: {claim}

Choose one verdict:
- SUPPORTED: you are confident the claim is true.
- REFUTED: you are confident the claim is false.
- NOT_ENOUGH_INFO: you are not sure either way.

Give a one-sentence reason, then the verdict."""


def build_prompt(claim, abstracts, max_words=300):
    """abstracts is a list of {"title", "text"}; an empty list means closed-book."""
    if not abstracts:
        return CLOSED_BOOK.format(claim=claim)
    blocks = []
    for number, abstract in enumerate(abstracts, start=1):
        text = " ".join(abstract["text"].split()[:max_words])
        blocks.append(f"Abstract {number}: {abstract['title']}\n{text}")
    return OPEN_BOOK.format(claim=claim, abstracts="\n\n".join(blocks))


def parse_verdict(raw):
    """Return (verdict, reason). An unreadable reply counts as NOT_ENOUGH_INFO."""
    try:
        reply = json.loads(raw)
        verdict, reason = reply["verdict"], reply["reason"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "NOT_ENOUGH_INFO", PARSE_FAILED
    if verdict not in LABELS:
        return "NOT_ENOUGH_INFO", PARSE_FAILED
    return verdict, reason


def judge(claim, abstracts):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": build_prompt(claim, abstracts)}],
            "format": REPLY_SCHEMA,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096},
        },
        timeout=600,
    )
    response.raise_for_status()
    return parse_verdict(response.json()["message"]["content"])
