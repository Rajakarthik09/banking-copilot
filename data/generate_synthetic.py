"""Generate the synthetic core-banking dataset and load it into Postgres.

Deterministic (seed=42) and idempotent: re-running truncates and reloads the same rows.
All dates are relative to settings.as_of_date, never to "today".

Run:  python data/generate_synthetic.py
"""

import random
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import psycopg
from faker import Faker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings  # noqa: E402

SEED = 42
N_CUSTOMERS = 1_000
N_LOANS = 1_500
N_TXNS = 50_000
PAST_DUE_SHARE = 0.12  # share of loans with days_past_due > 30

RMS = [
    ("RM-1", "Anna Weber", "Berlin"),
    ("RM-2", "Lukas Schmidt", "Hamburg"),
    ("RM-3", "Sofia Becker", "Munich"),
    ("RM-4", "Jonas Fischer", "Leipzig"),
    ("RM-5", "Mia Hoffmann", "Frankfurt"),
]

# Must match data/policies/aml_transaction_monitoring.md.
HIGH_RISK_COUNTRIES = ["IR", "KP", "MM", "SY", "YE"]
NORMAL_COUNTRIES = ["DE"] * 14 + ["AT", "NL", "FR", "PL", "IT", "ES", "CH", "US", "GB"]
STRUCTURING_MIN, STRUCTURING_MAX = 9_000, 9_999  # AML-02: just below the 10,000 EUR threshold
VELOCITY_BURST = 6  # AML-03: >= 5 transactions within 24 hours
N_AML01, N_AML02_GROUPS, N_AML03_GROUPS = 200, 50, 25  # 200 + 150 + 150 = 500 flagged (1%)

CHANNELS = ["card", "sepa", "online", "cash"]
# Controlled German phone formats so the PII regex can be tested against all of them.
PHONE_FORMATS = ["+49 15# ########", "015# ########", "+49 30 #######", "030 ########", "+49 (0)89 #######"]


def build(rng: random.Random, fake: Faker) -> dict[str, list[tuple]]:
    as_of = get_settings().as_of_date

    customers = []
    for i in range(1, N_CUSTOMERS + 1):
        rm_id, _, region = RMS[(i - 1) % len(RMS)]
        segment = "sme" if rng.random() < 0.25 else "retail"
        name = fake.company() if segment == "sme" else fake.name()
        kyc = as_of - timedelta(days=rng.randint(30, 36 * 30))
        phone = fake.numerify(rng.choice(PHONE_FORMATS))
        customers.append(
            (f"C{i:04d}", rm_id, name, segment, region, fake.iban(), fake.email(), phone, kyc)
        )
    segment_of = {c[0]: c[3] for c in customers}

    loans = []
    loan_customers = rng.choices([c[0] for c in customers], k=N_LOANS)
    n_past_due = round(N_LOANS * PAST_DUE_SHARE)
    past_due_idx = set(rng.sample(range(N_LOANS), n_past_due))
    for i, cid in enumerate(loan_customers):
        if segment_of[cid] == "sme":
            product = rng.choice(["sme_term", "sme_term", "sme_term", "mortgage"])
        else:
            product = rng.choice(["mortgage", "consumer", "consumer"])
        principal = {
            "mortgage": rng.randint(80, 600) * 1_000,
            "sme_term": rng.randint(25, 500) * 1_000,
            "consumer": rng.randint(2, 40) * 1_000,
        }[product]
        outstanding = round(principal * rng.uniform(0.2, 0.98), 2)
        rate = round(rng.uniform(2.5, 9.5), 2)
        if i in past_due_idx:
            dpd = rng.randint(31, 180)
        else:
            dpd = rng.randint(1, 30) if rng.random() < 0.08 else 0
        if dpd > 90:
            tier = "D"
        elif dpd > 30:
            tier = rng.choice(["C", "D"])
        else:
            tier = rng.choices(["A", "B", "C"], weights=[50, 35, 15])[0]
        status = "default" if dpd > 90 else "past_due" if dpd > 0 else "current"
        loans.append((f"L{i + 1:05d}", cid, product, principal, outstanding, rate, dpd, tier, status))

    start = datetime.combine(as_of - timedelta(days=365), time())
    span_s = 364 * 24 * 3600
    cids = [c[0] for c in customers]

    def normal_amount() -> float:
        amt = round(min(rng.lognormvariate(5.0, 1.2), 50_000), 2)
        # Keep non-flagged amounts out of the structuring band so AML-02 stays meaningful.
        return 8_999.0 if STRUCTURING_MIN <= amt <= STRUCTURING_MAX + 1 else amt

    raw: list[list] = []  # [customer_id, ts, amount, currency, country, channel, flagged, flag_rule]
    for _ in range(N_AML01):
        ts = start + timedelta(seconds=rng.randint(0, span_s))
        raw.append([rng.choice(cids), ts, round(rng.uniform(500, 40_000), 2), "EUR",
                    rng.choice(HIGH_RISK_COUNTRIES), "sepa", True, "AML-01 high-risk country"])
    for _ in range(N_AML02_GROUPS):  # 3 cash deposits just under 10k within 7 days
        cid, t0 = rng.choice(cids), start + timedelta(seconds=rng.randint(0, span_s - 7 * 86400))
        for _ in range(3):
            ts = t0 + timedelta(seconds=rng.randint(0, 7 * 86400))
            raw.append([cid, ts, float(rng.randint(STRUCTURING_MIN, STRUCTURING_MAX)), "EUR", "DE",
                        "cash", True, "AML-02 structuring < 10k"])
    for _ in range(N_AML03_GROUPS):  # 6 transactions within 24 hours
        cid, t0 = rng.choice(cids), start + timedelta(seconds=rng.randint(0, span_s - 86400))
        for _ in range(VELOCITY_BURST):
            ts = t0 + timedelta(seconds=rng.randint(0, 86400 - 1))
            raw.append([cid, ts, normal_amount(), "EUR", rng.choice(NORMAL_COUNTRIES),
                        rng.choice(["card", "online"]), True, "AML-03 velocity"])
    while len(raw) < N_TXNS:
        ts = start + timedelta(seconds=rng.randint(0, span_s))
        currency = "EUR" if rng.random() < 0.95 else rng.choice(["USD", "CHF", "GBP"])
        raw.append([rng.choice(cids), ts, normal_amount(), currency, rng.choice(NORMAL_COUNTRIES),
                    rng.choice(CHANNELS), False, None])
    raw.sort(key=lambda r: r[1])  # IDs in time order, so flagged txns are spread through the ID range
    txns = [(f"T{i + 1:06d}", *r) for i, r in enumerate(raw)]

    return {"relationship_managers": RMS, "customers": customers, "loans": loans, "transactions": txns}


COLUMNS = {
    "relationship_managers": "rm_id, name, region",
    "customers": "customer_id, rm_id, name, segment, region, iban, email, phone, kyc_last_review",
    "loans": "loan_id, customer_id, product, principal, outstanding, rate, days_past_due, risk_tier, status",
    "transactions": "txn_id, customer_id, ts, amount, currency, counterparty_country, channel, flagged, flag_rule",
}


def main() -> None:
    rng = random.Random(SEED)
    fake = Faker("de_DE")
    fake.seed_instance(SEED)
    data = build(rng, fake)

    schema = (Path(__file__).parent / "schema.sql").read_text()
    with psycopg.connect(get_settings().db_url, autocommit=False) as conn:
        conn.execute(schema)
        conn.execute("TRUNCATE transactions, loans, customers, relationship_managers CASCADE")
        for table, cols in COLUMNS.items():
            with conn.cursor().copy(f"COPY {table} ({cols}) FROM STDIN") as copy:
                for row in data[table]:
                    copy.write_row(row)
        conn.commit()
        for table in COLUMNS:
            n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            expected = len(data[table])
            status = "OK" if n == expected else "MISMATCH"
            print(f"{table:<22} {n:>6} rows (expected {expected}) {status}")
            if n != expected:
                sys.exit(1)


if __name__ == "__main__":
    main()
