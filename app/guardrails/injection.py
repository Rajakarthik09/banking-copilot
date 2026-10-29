"""Heuristic prompt-injection check for retrieved text.

Retrieved documents are data, not instructions. Chunks that look like they are trying
to instruct the model are dropped before generation.
"""

import re

_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |any )?(the )?(previous|prior|above|earlier|your) (instructions|rules)",
    r"forget (all |your )?(previous |prior )?instructions",
    r"system prompt",
    r"you are now\b",
    r"\bact as (an? )?(admin|administrator|developer|system)",
    r"reveal (your|the) (instructions|prompt|rules)",
    r"(list|show|print|dump) all (customer )?(ibans?|passwords?|customers?|credentials)",
    r"</?(system|assistant)>",
]
_RE = re.compile("|".join(_PATTERNS), re.I)


def find_injection(text: str) -> str | None:
    """Return the matched suspicious phrase, or None if the text looks clean."""
    m = _RE.search(text or "")
    return m.group(0) if m else None


def filter_chunks(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split chunks into (clean, dropped). Dropped chunks carry the matched phrase."""
    clean, dropped = [], []
    for c in chunks:
        hit = find_injection(c.get("content", ""))
        if hit:
            dropped.append({**c, "injection_match": hit})
        else:
            clean.append(c)
    return clean, dropped
