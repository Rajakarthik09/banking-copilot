import pytest

from app.rag.retrieve import check_embedding_model, format_excerpt, retrieve, rrf_merge


def _c(cid):
    return {"chunk_id": cid, "doc": "d", "section": "1. S", "content": "x"}


def test_rrf_rewards_agreement_between_rankings():
    vector = [_c("a"), _c("b"), _c("c")]
    fts = [_c("b")]
    order = [x["chunk_id"] for x in rrf_merge(vector, fts)]
    assert order == ["b", "a", "c"]  # 1/62 + 1/61 (found by both) beats 1/61 (top of one list only)


def test_format_excerpt_tags_every_line():
    chunk = {"doc": "kyc_refresh", "section": "2. Review Frequency",
             "content": "KYC Refresh Policy - 2. Review Frequency\n1. Every 12 months.\n2. Every 24 months."}
    out = format_excerpt(chunk).splitlines()
    assert out[0] == "[kyc_refresh §2] Review Frequency"
    assert all(line.endswith("[kyc_refresh §2]") for line in out[1:])


def test_long_natural_question_still_gets_keyword_hits(db):
    r = retrieve("Could you please tell me how many overdraft fee waivers one customer may receive per year?")
    assert r.top_fts > 0 and r.chunks[0]["doc"] == "overdraft_fee_waiver"


def test_embedding_model_mismatch_is_loud(db):
    with pytest.raises(RuntimeError, match="re-run"):
        check_embedding_model("some-other-model/768")
