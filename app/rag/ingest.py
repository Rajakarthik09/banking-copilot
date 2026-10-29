"""Chunk policy markdown by `##` section, embed, and store in pgvector.

Idempotent: deletes and reloads all chunks. Files starting with "_" are test fixtures
and are skipped unless --include-test is given.

Run:  python -m app.rag.ingest
"""

import argparse
import logging
import re
from pathlib import Path

from app.db import get_conn
from app.llm import get_embedder

log = logging.getLogger(__name__)
POLICY_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"
_SECTION = re.compile(r"^##\s+(\d+)\.?\s*(.*)$")


def chunk_markdown(doc: str, text: str) -> list[dict]:
    """One chunk per `## N. Title` section. The doc title is prepended for embedding context."""
    title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), doc)
    chunks, current, body = [], None, []

    def flush():
        if current and "".join(body).strip():
            num, name = current
            chunks.append({
                "chunk_id": f"{doc}§{num}",
                "doc": doc,
                "section": f"{num}. {name}".strip(),
                "content": f"{title} - {num}. {name}\n" + "\n".join(body).strip(),
            })

    for line in text.splitlines():
        m = _SECTION.match(line)
        if m:
            flush()
            current, body = (m.group(1), m.group(2).strip()), []
        elif current:
            body.append(line)
    flush()
    return chunks


def load_chunks(include_test: bool = False) -> list[dict]:
    chunks = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        if path.name.startswith("_") and not include_test:
            continue
        chunks.extend(chunk_markdown(path.stem, path.read_text()))
    return chunks


def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


def ingest(include_test: bool = False) -> int:
    chunks = load_chunks(include_test)
    embedder = get_embedder()
    vectors = embedder.embed([c["content"] for c in chunks], kind="document")
    with get_conn() as conn, conn.transaction():
        conn.execute("DELETE FROM policy_chunks")
        for c, v in zip(chunks, vectors, strict=True):
            conn.execute(
                "INSERT INTO policy_chunks (id, doc, section, content, embedding, embed_model) "
                "VALUES (%s, %s, %s, %s, %s::vector, %s)",
                (c["chunk_id"], c["doc"], c["section"], c["content"], to_pgvector(v), embedder.name),
            )
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true", help="also ingest _*.md fixtures")
    n = ingest(ap.parse_args().include_test)
    log.info("ingested %d chunks", n)
