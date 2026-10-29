"""Typed, read-only banking tools with role-based access control.

Every tool takes `rm_id` as its first argument. It is injected by the server from the
authenticated request (or the MCP server's RM_ID), never chosen by the LLM: the tool
schemas in registry.py don't contain it, and argument models forbid extra fields.
SQL is parameterized; the LLM never writes SQL.

A customer or transaction that doesn't exist and one that belongs to another RM give the
same AccessDenied, so the tools don't reveal which IDs exist.
"""

import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import get_settings
from app.db import fetch_all, fetch_one
from app.guardrails.pii import mask_pii

Segment = Literal["retail", "sme"]
RiskTier = Literal["A", "B", "C", "D"]
_RM_RE = re.compile(r"^RM-\d+$")


class AccessDenied(Exception):
    pass


def _check_rm(rm_id: str) -> None:
    if not isinstance(rm_id, str) or not _RM_RE.match(rm_id):
        raise AccessDenied("unknown relationship manager")


def _normalise_id(value: str | int, prefix: str, width: int) -> str:
    """Accept 'C0042', 'c42', '42', 'customer 42' -> 'C0042' (small models drop prefixes)."""
    digits = re.sub(r"\D", "", str(value))
    if not digits or len(digits) > width:
        raise ValueError(f"expected an ID like {prefix}{'1'.zfill(width)}")
    return f"{prefix}{int(digits):0{width}d}"


# ---------- argument models (what the LLM may send; rm_id is never one of them) ----------

class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListPastDueLoansArgs(_Args):
    min_dpd: int = Field(30, ge=0, le=365, description="Return loans MORE than this many days past due")
    segment: Segment | None = Field(None, description="Customer segment filter")


class PortfolioExposureArgs(_Args):
    segment: Segment | None = Field(None, description="Customer segment filter")
    risk_tier: RiskTier | None = Field(None, description="Loan risk tier filter")


class CustomerProfileArgs(_Args):
    customer_id: str = Field(description="Customer ID, e.g. C0042")

    @field_validator("customer_id", mode="before")
    @classmethod
    def _norm(cls, v):
        return _normalise_id(v, "C", 4)


class TransactionArgs(_Args):
    txn_id: str = Field(description="Transaction ID, e.g. T000123")

    @field_validator("txn_id", mode="before")
    @classmethod
    def _norm(cls, v):
        return _normalise_id(v, "T", 6)


class KycDueArgs(_Args):
    months: int = Field(24, ge=1, le=120, description="Months since the last KYC review")


# ---------- output models ----------

class LoanSummary(BaseModel):
    loan_id: str
    customer_id: str
    customer_name: str
    segment: str
    product: str
    outstanding: float
    days_past_due: int
    risk_tier: str


class PastDueLoans(BaseModel):
    currency: str = "EUR"
    min_dpd: int
    segment: str | None
    count: int
    total_outstanding: float
    loans: list[LoanSummary]
    truncated: bool = Field(description="True if only the first rows are listed; count/total cover all")


class ExposureSummary(BaseModel):
    currency: str = "EUR"
    segment: str | None
    risk_tier: str | None
    loan_count: int
    customer_count: int
    total_outstanding: float
    total_principal: float
    outstanding_by_risk_tier: dict[str, float]
    outstanding_by_product: dict[str, float]


class CustomerProfile(BaseModel):
    currency: str = "EUR"
    customer_id: str
    name: str
    segment: str
    region: str
    iban: str
    email: str
    phone: str
    kyc_last_review: date
    loan_count: int
    total_outstanding: float
    max_days_past_due: int
    flagged_transaction_count: int


class Transaction(BaseModel):
    txn_id: str
    customer_id: str
    ts: datetime
    amount: float
    currency: str
    counterparty_country: str
    channel: str
    flagged: bool
    flag_rule: str | None


class CustomerRef(BaseModel):
    customer_id: str
    name: str
    segment: str
    kyc_last_review: date


class KycDueList(BaseModel):
    months: int
    as_of: date
    count: int
    customers: list[CustomerRef]
    truncated: bool


# ---------- tools ----------

def list_past_due_loans(rm_id: str, min_dpd: int = 30, segment: Segment | None = None) -> PastDueLoans:
    _check_rm(rm_id)
    rows = fetch_all(
        """
        SELECT l.loan_id, l.customer_id, c.name AS customer_name, c.segment, l.product,
               l.outstanding::float AS outstanding, l.days_past_due, l.risk_tier
        FROM loans l JOIN customers c USING (customer_id)
        WHERE c.rm_id = %(rm)s AND l.days_past_due > %(dpd)s
          AND (%(seg)s::text IS NULL OR c.segment = %(seg)s)
        ORDER BY l.days_past_due DESC, l.outstanding DESC, l.loan_id
        """,
        {"rm": rm_id, "dpd": min_dpd, "seg": segment},
    )
    limit = get_settings().tool_row_limit
    return PastDueLoans(
        min_dpd=min_dpd,
        segment=segment,
        count=len(rows),
        total_outstanding=round(sum(r["outstanding"] for r in rows), 2),
        loans=[LoanSummary(**r) for r in rows[:limit]],
        truncated=len(rows) > limit,
    )


def get_portfolio_exposure(rm_id: str, segment: Segment | None = None,
                           risk_tier: RiskTier | None = None) -> ExposureSummary:
    """Exposure = sum of outstanding balances of the RM's loans matching the filters."""
    _check_rm(rm_id)
    rows = fetch_all(
        """
        SELECT l.customer_id, l.product, l.risk_tier, l.outstanding::float AS outstanding,
               l.principal::float AS principal
        FROM loans l JOIN customers c USING (customer_id)
        WHERE c.rm_id = %(rm)s
          AND (%(seg)s::text IS NULL OR c.segment = %(seg)s)
          AND (%(tier)s::text IS NULL OR l.risk_tier = %(tier)s)
        """,
        {"rm": rm_id, "seg": segment, "tier": risk_tier},
    )
    by_tier: dict[str, float] = {}
    by_product: dict[str, float] = {}
    for r in rows:
        by_tier[r["risk_tier"]] = by_tier.get(r["risk_tier"], 0.0) + r["outstanding"]
        by_product[r["product"]] = by_product.get(r["product"], 0.0) + r["outstanding"]
    return ExposureSummary(
        segment=segment,
        risk_tier=risk_tier,
        loan_count=len(rows),
        customer_count=len({r["customer_id"] for r in rows}),
        total_outstanding=round(sum(r["outstanding"] for r in rows), 2),
        total_principal=round(sum(r["principal"] for r in rows), 2),
        outstanding_by_risk_tier={k: round(v, 2) for k, v in sorted(by_tier.items())},
        outstanding_by_product={k: round(v, 2) for k, v in sorted(by_product.items())},
    )


def get_customer_profile(rm_id: str, customer_id: str) -> CustomerProfile:
    _check_rm(rm_id)
    row = fetch_one(
        """
        SELECT c.customer_id, c.name, c.segment, c.region, c.iban, c.email, c.phone, c.kyc_last_review,
               (SELECT count(*) FROM loans l WHERE l.customer_id = c.customer_id) AS loan_count,
               (SELECT coalesce(sum(outstanding), 0)::float FROM loans l
                 WHERE l.customer_id = c.customer_id) AS total_outstanding,
               (SELECT coalesce(max(days_past_due), 0) FROM loans l
                 WHERE l.customer_id = c.customer_id) AS max_days_past_due,
               (SELECT count(*) FROM transactions t
                 WHERE t.customer_id = c.customer_id AND t.flagged) AS flagged_transaction_count
        FROM customers c
        WHERE c.customer_id = %(cid)s AND c.rm_id = %(rm)s
        """,
        {"cid": customer_id, "rm": rm_id},
    )
    if row is None:
        raise AccessDenied(f"customer {customer_id} not found or not in your portfolio")
    for key in ("iban", "email", "phone"):
        row[key] = mask_pii(row[key])
    return CustomerProfile(**row)


def get_transaction(rm_id: str, txn_id: str) -> Transaction:
    _check_rm(rm_id)
    row = fetch_one(
        """
        SELECT t.txn_id, t.customer_id, t.ts, t.amount::float AS amount, t.currency,
               t.counterparty_country, t.channel, t.flagged, t.flag_rule
        FROM transactions t JOIN customers c USING (customer_id)
        WHERE t.txn_id = %(tid)s AND c.rm_id = %(rm)s
        """,
        {"tid": txn_id, "rm": rm_id},
    )
    if row is None:
        raise AccessDenied(f"transaction {txn_id} not found or not in your portfolio")
    return Transaction(**row)


def customers_due_for_kyc(rm_id: str, months: int = 24) -> KycDueList:
    """Customers whose last KYC review is at least `months` months before the as-of date."""
    _check_rm(rm_id)
    as_of = get_settings().as_of_date
    rows = fetch_all(
        """
        SELECT customer_id, name, segment, kyc_last_review
        FROM customers
        WHERE rm_id = %(rm)s AND kyc_last_review <= %(as_of)s::date - make_interval(months => %(m)s)
        ORDER BY kyc_last_review, customer_id
        """,
        {"rm": rm_id, "as_of": as_of, "m": months},
    )
    limit = get_settings().tool_row_limit
    return KycDueList(
        months=months,
        as_of=as_of,
        count=len(rows),
        customers=[CustomerRef(**r) for r in rows[:limit]],
        truncated=len(rows) > limit,
    )
