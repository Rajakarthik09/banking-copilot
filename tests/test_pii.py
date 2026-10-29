import json

import pytest
from faker import Faker

from app.guardrails.pii import mask_pii

fake = Faker("de_DE")
fake.seed_instance(7)


def test_1000_faker_de_phones_fully_masked():
    misses = [v for v in (fake.phone_number() for _ in range(1000)) if mask_pii(v).strip() != "[PHONE]"]
    assert misses == []


def test_generator_phone_formats_masked():
    for v in ["+49 151 23456789", "0151 23456789", "+49 30 1234567", "030 12345678", "+49 (0)89 1234567"]:
        assert mask_pii(v) == "[PHONE]", v


def test_300_faker_ibans_and_emails_masked():
    assert [v for v in (fake.iban() for _ in range(300)) if mask_pii(v) != "[IBAN]"] == []
    assert [v for v in (fake.email() for _ in range(300)) if mask_pii(v) != "[EMAIL]"] == []


def test_spaced_iban_and_email_masked():
    assert mask_pii("IBAN DE89 3704 0044 0532 0130 00, mail anna.weber@example.org") == \
        "IBAN [IBAN], mail [EMAIL]"


@pytest.mark.parametrize("value", [
    "C0042", "L00017", "T000123", "RM-1", "AML-02", "12,500.00 EUR", "1,234,567.89", "9,950", "0.5%",
    "2026-10-31", "2026-10-31T10:22:00", "10:22:00", "180 days", "outstanding 450000.00", "rate 4.25",
    '{"amount": 9123.0, "days_past_due": 45}', "954342.98", "12-month period",
])
def test_business_values_not_masked(value):
    assert mask_pii(value) == value


def test_tool_json_masked_but_numbers_intact():
    blob = json.dumps({"iban": "DE92960013389083863794", "email": "x@y.de", "phone": "030 12345678",
                       "total_outstanding": 954342.98, "kyc_last_review": "2026-08-11", "customer_id": "C0001"})
    masked = json.loads(mask_pii(blob))
    assert masked["iban"] == "[IBAN]" and masked["email"] == "[EMAIL]" and masked["phone"] == "[PHONE]"
    assert masked["total_outstanding"] == 954342.98 and masked["kyc_last_review"] == "2026-08-11"
    assert masked["customer_id"] == "C0001"


def test_question_and_audit_are_masked(db, scripted):
    from app.db import fetch_one
    from app.llm import ChatResult
    from app.orchestrator import answer_question

    q = "Customer with IBAN DE89370400440532013000 and phone 0151 23456789 asked about fees"
    llm = scripted([ChatResult(content="I can only help with bank policies and your portfolio.")])
    answer_question(q, "RM-1", "prompt_only", verify=False, llm=llm)
    sent = llm.calls[0][-1]["content"]
    assert "DE89" not in sent and "0151" not in sent and "[IBAN]" in sent
    row = fetch_one("SELECT question_masked FROM audit_log ORDER BY id DESC LIMIT 1")
    assert "[IBAN]" in row["question_masked"] and "[PHONE]" in row["question_masked"]
