"""One audit_log row per /chat request. Questions and answers are PII-masked before storage.

Audit is fail-closed: if the row can't be written, the request fails (a bank must be able
to show who asked what, with which tools).
"""

import json

from app.db import get_conn
from app.guardrails.pii import mask_pii


def write_audit(*, rm_id: str, mode: str, question: str, tools_called: list[dict], sources: list[dict],
                answer: str, verified: bool | None, refused: bool, latency_ms: int, tokens: int) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO audit_log (rm_id, mode, question_masked, tools_called, sources, answer,
                                      verified, refused, latency_ms, tokens)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (rm_id, mode, mask_pii(question), json.dumps(tools_called, default=str),
             json.dumps(sources), mask_pii(answer), verified, refused, latency_ms, tokens),
        )
