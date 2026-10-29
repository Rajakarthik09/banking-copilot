"""Hybrid retrieval: pgvector cosine top-k + Postgres full-text top-k, merged by reciprocal-rank fusion.

RRF score = sum over both rankings of 1 / (RRF_K + rank). It only uses ranks, so it can't say
whether the best match is actually relevant; that decision uses the top cosine similarity.
"""

from dataclasses import dataclass, field

from app.config import get_settings
from app.db import fetch_all
from app.guardrails.injection import filter_chunks
from app.llm import get_embedder
from app.rag.ingest import to_pgvector

RRF_K = 60

_VECTOR_SQL = """
SELECT id AS chunk_id, doc, section, content, 1 - (embedding <=> %(v)s::vector) AS sim
FROM policy_chunks ORDER BY embedding <=> %(v)s::vector LIMIT %(k)s
"""

# Natural questions have many words; plainto_tsquery would AND them all and match nothing.
# Build an OR query from the stemmed lexemes instead.
_FTS_SQL = """
WITH q AS (
  SELECT to_tsquery('simple', array_to_string(array(
    SELECT quote_literal(l) FROM unnest(tsvector_to_array(to_tsvector('english', %(q)s))) l), ' | ')) AS q
)
SELECT id AS chunk_id, doc, section, content, ts_rank(tsv, q.q) AS rank
FROM policy_chunks, q WHERE tsv @@ q.q ORDER BY rank DESC LIMIT %(k)s
"""


@dataclass
class Retrieval:
    chunks: list[dict]              # fused, injection-filtered, best first
    top_sim: float                  # best cosine similarity among vector hits
    top_fts: float = 0.0            # best full-text ts_rank (0 if no keyword match)
    dropped: list[dict] = field(default_factory=list)  # chunks removed by the injection check

    @property
    def weak(self) -> bool:
        s = get_settings()
        if not self.chunks:
            return True
        if self.top_sim >= s.rag_min_sim:
            return False
        return not (self.top_sim >= s.rag_min_sim_with_keyword and self.top_fts >= s.rag_min_fts)


def source_tag(chunk: dict) -> str:
    return f"[{chunk['doc']} §{chunk['section'].split('.')[0]}]"


def format_excerpt(chunk: dict) -> str:
    """Prompt form of a chunk: tag + section title, then every line ending with its tag.

    Small models copy well but attribute poorly across a long context; with the tag on
    each rule line, a copied rule carries its own correct citation.
    """
    tag = source_tag(chunk)
    title = chunk["section"].split(".", 1)[-1].strip()
    lines = [ln.strip() for ln in chunk["content"].split("\n")[1:] if ln.strip()]
    return f"{tag} {title}\n" + "\n".join(f"{ln} {tag}" for ln in lines)


def rrf_merge(*rankings: list[dict], k: int = RRF_K) -> list[dict]:
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(cid, {key: item[key] for key in ("chunk_id", "doc", "section", "content")})
    ordered = sorted(scores, key=lambda c: scores[c], reverse=True)
    return [{**by_id[c], "rrf": round(scores[c], 5)} for c in ordered]


def check_embedding_model(embedder_name: str) -> None:
    """Vectors from different models live in different spaces; comparing them fails silently."""
    stored = {r["embed_model"] for r in fetch_all("SELECT DISTINCT embed_model FROM policy_chunks")}
    if not stored:
        raise RuntimeError("policy_chunks is empty: run `python -m app.rag.ingest` first")
    if stored != {embedder_name}:
        raise RuntimeError(f"policy_chunks were embedded with {sorted(stored)} but queries use "
                           f"{embedder_name!r}: re-run `python -m app.rag.ingest`")


_checked: set[str] = set()


def retrieve(question: str, top_k: int | None = None) -> Retrieval:
    k = top_k or get_settings().rag_top_k
    embedder = get_embedder()
    if embedder.name not in _checked:
        check_embedding_model(embedder.name)
        _checked.add(embedder.name)
    vec = to_pgvector(embedder.embed([question], kind="query")[0])
    vector_hits = fetch_all(_VECTOR_SQL, {"v": vec, "k": k})
    fts_hits = fetch_all(_FTS_SQL, {"q": question, "k": k})
    top_sim = max((float(h["sim"]) for h in vector_hits), default=0.0)
    top_fts = max((float(h["rank"]) for h in fts_hits), default=0.0)
    fused = rrf_merge(vector_hits, fts_hits)[:k]
    clean, dropped = filter_chunks(fused)
    return Retrieval(chunks=clean, top_sim=round(top_sim, 4), top_fts=round(top_fts, 4), dropped=dropped)
