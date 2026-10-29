-- Idempotent: run by Postgres init and again by data/generate_synthetic.py.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS relationship_managers (
    rm_id   text PRIMARY KEY,
    name    text NOT NULL,
    region  text NOT NULL
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id      text PRIMARY KEY,
    rm_id            text NOT NULL REFERENCES relationship_managers(rm_id),
    name             text NOT NULL,
    segment          text NOT NULL CHECK (segment IN ('retail', 'sme')),
    region           text NOT NULL,
    iban             text NOT NULL,
    email            text NOT NULL,
    phone            text NOT NULL,
    kyc_last_review  date NOT NULL
);
CREATE INDEX IF NOT EXISTS customers_rm_idx ON customers(rm_id);

CREATE TABLE IF NOT EXISTS loans (
    loan_id        text PRIMARY KEY,
    customer_id    text NOT NULL REFERENCES customers(customer_id),
    product        text NOT NULL CHECK (product IN ('mortgage', 'sme_term', 'consumer')),
    principal      numeric(14, 2) NOT NULL,
    outstanding    numeric(14, 2) NOT NULL,
    rate           numeric(5, 2) NOT NULL,
    days_past_due  integer NOT NULL,
    risk_tier      text NOT NULL CHECK (risk_tier IN ('A', 'B', 'C', 'D')),
    status         text NOT NULL
);
CREATE INDEX IF NOT EXISTS loans_customer_idx ON loans(customer_id);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id                text PRIMARY KEY,
    customer_id           text NOT NULL REFERENCES customers(customer_id),
    ts                    timestamp NOT NULL,
    amount                numeric(14, 2) NOT NULL,
    currency              text NOT NULL,
    counterparty_country  text NOT NULL,
    channel               text NOT NULL,
    flagged               boolean NOT NULL DEFAULT false,
    flag_rule             text
);
CREATE INDEX IF NOT EXISTS transactions_customer_idx ON transactions(customer_id);

CREATE TABLE IF NOT EXISTS policy_chunks (
    id          text PRIMARY KEY,          -- e.g. overdraft_fee_waiver§2
    doc         text NOT NULL,             -- file stem, e.g. overdraft_fee_waiver
    section     text NOT NULL,             -- e.g. "2. Eligibility"
    content     text NOT NULL,
    embedding   vector(768) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', section || ' ' || content)) STORED
);
-- Which embedding model produced the vectors; retrieval refuses to mix models.
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS embed_model text NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS policy_chunks_tsv_idx ON policy_chunks USING gin(tsv);

CREATE TABLE IF NOT EXISTS audit_log (
    id               bigserial PRIMARY KEY,
    ts               timestamptz NOT NULL DEFAULT now(),
    rm_id            text NOT NULL,
    mode             text NOT NULL,
    question_masked  text NOT NULL,
    tools_called     jsonb NOT NULL DEFAULT '[]',
    sources          jsonb NOT NULL DEFAULT '[]',
    answer           text NOT NULL,
    verified         boolean,
    refused          boolean NOT NULL DEFAULT false,
    latency_ms       integer NOT NULL,
    tokens           integer
);
