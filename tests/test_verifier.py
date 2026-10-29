from app.verifier import NOTE, extract_numbers, split_sentences, verify

TOOL = ['{"currency": "EUR", "count": 8, "total_outstanding": 954342.98, '
        '"loans": [{"loan_id": "L00017", "days_past_due": 45, "outstanding": 42487.8}]}']
CHUNKS = {
    ("overdraft_fee_waiver", "2"): "Overdraft Fee Waiver Policy - 2. Eligibility\n1. A customer may receive "
                                   "at most 2 overdraft fee waivers in any rolling 12-month period.",
    ("overdraft_fee_waiver", "3"): "Overdraft Fee Waiver Policy - 3. Amount Limits\n1. A single waiver covers "
                                   "at most one overdraft fee of up to 25 EUR.",
}


def test_grounded_answer_passes_unchanged():
    a = "You have 8 SME loans more than 30 days past due, total 954,342.98 EUR. L00017 is 45 days past due."
    r = verify(a, TOOL, {}, question="Which SME loans are more than 30 days past due?")
    assert r.verified and r.answer == a


def test_hallucinated_number_removed_and_flagged():
    r = verify("You have 9 loans past due. The total is 954,342.98 EUR.", TOOL, {})
    assert not r.verified
    assert r.removed == ["You have 9 loans past due."]
    assert "954,342.98" in r.answer and NOTE in r.answer


def test_rounded_and_unit_values_supported():
    for a in ["Total is about 954,343 EUR.", "Total is €0.95 million.", "Total is 954.3k EUR.",
              "Loan L00017 has 42,487.80 EUR outstanding."]:
        assert verify(a, TOOL, {}).verified, a


def test_wrong_rounding_rejected():
    assert not verify("Total is €0.97 million.", TOOL, {}).verified


def test_numbers_from_question_ids_and_list_markers_ignored():
    a = "1. Loan L00017 (customer C0042, RM-1) is flagged under AML-02 and is more than 30 days late."
    assert verify(a, TOOL, {}, question="loans over 30 days").verified


def test_valid_citation_in_any_spelling():
    for cite in ["[overdraft_fee_waiver §2]", "[Overdraft Fee Waiver Policy §2]", "[overdraft_fee_waiver.md §2]",
                 "(overdraft_fee_waiver §2)", "[overdraft_fee_waiver §2. Eligibility]"]:
        assert verify(f"At most 2 waivers per rolling 12-month period {cite}.", [], CHUNKS).verified, cite


def test_citation_to_unretrieved_chunk_removed():
    r = verify("At most 2 waivers per 12 months [overdraft_fee_waiver §5].", [], CHUNKS)
    assert not r.verified and r.bad_citations == ["[overdraft_fee_waiver §5]"]


def test_misattributed_citation_removed():
    # The number is real (in §2) but the sentence cites §3, which doesn't contain it.
    r = verify("A customer may get at most 2 waivers in 12 months. [overdraft_fee_waiver §3]", [], CHUNKS)
    assert not r.verified and r.unsupported_numbers


def test_trailing_citation_stays_with_sentence():
    parts = split_sentences("At most 2 waivers. [overdraft_fee_waiver §2] Next sentence here.")
    assert parts == ["At most 2 waivers. [overdraft_fee_waiver §2]", "Next sentence here."]


def test_foreign_currency_hallucination_removed():
    r = verify("Total exposure is $954,342.98.", TOOL, {})
    assert not r.verified


def test_dates_checked_in_any_format():
    ev = ['{"kyc_last_review": "2026-08-11"}']
    assert verify("Last review: 11 August 2026.", ev, {}).verified
    assert verify("Last review: 2026-08-11.", ev, {}).verified
    assert not verify("Last review: 12 August 2026.", ev, {}).verified


def test_everything_removed_gives_fallback():
    r = verify("There are 17 loans.", TOOL, {})
    assert not r.verified and r.answer.startswith("I couldn't produce an answer")


def test_number_extraction_units():
    nums, _ = extract_numbers("12.5% rate, 1.2M, 3k and 9,950 EUR")
    assert [n.value for n in nums] == [12.5, 1_200_000, 3_000, 9_950]
