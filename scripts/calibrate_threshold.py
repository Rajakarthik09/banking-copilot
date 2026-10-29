"""Print top cosine similarity for in-scope vs out-of-scope questions and suggest RAG_MIN_SIM.

Run after ingest:  python scripts/calibrate_threshold.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rag.retrieve import retrieve  # noqa: E402

IN_SCOPE = [
    "How many fee waivers can a customer get per year?",
    "What is the maximum overdraft fee an RM can waive?",
    "How long can a payment holiday last under forbearance?",
    "How often must a high-risk customer have a KYC review?",
    "What happens if a KYC review is more than 60 days overdue?",
    "Which countries trigger AML-01?",
    "What counts as structuring?",
    "How fast must an AML alert be reviewed?",
    "When is an SME loan moved to tier D?",
    "What is the RM approval limit for new SME lending?",
    "How quickly must we acknowledge a complaint?",
    "How much goodwill compensation can an RM offer?",
    "How long are transaction records kept?",
    "When does an account become dormant?",
    "Can a customer close an account with a negative balance?",
    "Why was this transaction flagged for velocity?",
]
OUT_OF_SCOPE = [
    "What's the weather in Leipzig?",
    "Who won the football world cup in 2014?",
    "Write me a poem about the sea.",
    "What is the capital of Australia?",
    "How do I bake sourdough bread?",
    "Recommend a good laptop for programming.",
    "What is the stock price of Apple today?",
    "Translate 'good morning' into Spanish.",
    "How tall is Mount Everest?",
    "Tell me a joke.",
]


def main() -> None:
    ins = [(retrieve(q), q) for q in IN_SCOPE]
    outs = [(retrieve(q), q) for q in OUT_OF_SCOPE]
    for label, rows in (("IN", ins), ("OUT", outs)):
        for r, q in sorted(rows, key=lambda x: x[0].top_sim):
            gate = "refuse" if r.weak else "pass"
            print(f"{label:<4}sim={r.top_sim:.3f} fts={r.top_fts:.3f} gate={gate:<6} {q}")
    in_pass = sum(not r.weak for r, _ in ins)
    out_block = sum(r.weak for r, _ in outs)
    print(f"\ncurrent gate: {in_pass}/{len(ins)} in-scope pass, {out_block}/{len(outs)} out-of-scope refused")
    ins = [(r.top_sim, q) for r, q in ins]
    outs = [(r.top_sim, q) for r, q in outs]
    lo_in, hi_out = min(s for s, _ in ins), max(s for s, _ in outs)
    print(f"\nmin in-scope = {lo_in:.3f}   max out-of-scope = {hi_out:.3f}")
    if lo_in > hi_out:
        print(f"similarity alone separable; could use RAG_MIN_SIM = {(lo_in + hi_out) / 2:.2f}")
    else:
        print("similarity alone not separable -> combined vector + keyword gate, prompt refusal as 2nd layer")


if __name__ == "__main__":
    main()
