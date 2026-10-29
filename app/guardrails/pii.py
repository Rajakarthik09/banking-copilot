"""Regex PII masking for IBAN, email and phone numbers.

Applied to the user question, tool outputs (before they reach the LLM) and the audit log.
The patterns are deliberately anchored so that amounts, dates and internal IDs
(C0042, L00017, T000123, 12,500.00, 2026-10-31) are never masked; see tests/test_pii.py.
"""

import re

# Two letters, two check digits, then 11-30 alphanumerics, optionally grouped in fours.
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(
    r"""
    (?<![\w+])                                   # not glued to an ID or word
    (?:
        (?:\+|00)\d{1,3}[ ]?(?:\(0\)[ ]?)?\d{2,5}  # +49 30 / +49 (0)89 / +49(0)5182 / 0049 151
      | \(0\d{2,5}\)                               # (06343)
      | 0\d{2,5}                                   # 030 / 0151
    )
    [ \-/]?\d{3,9}(?:[ \-]\d{2,5})?              # subscriber number, optional extension group
    (?![\w])
    """,
    re.X,
)


def mask_pii(text: str) -> str:
    if not text:
        return text
    text = IBAN_RE.sub("[IBAN]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    return PHONE_RE.sub("[PHONE]", text)
