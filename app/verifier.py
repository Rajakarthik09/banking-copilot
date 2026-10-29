"""Grounding verifier: every number and citation in an answer must be backed by the evidence.

Evidence = tool results + retrieved policy chunks + the user's question (numbers the user
typed are not hallucinations). A sentence with an unsupported number or an invalid
citation is removed and the answer is marked unverified.

Known limits (by design, see docs/decisions.md): numbers the LLM derives itself
(a sum it computed, a percentage) and numbers written as words ("twice") are not checked
as such; derived values fail unless the tools already return them, which is why the
tools return pre-computed counts and totals.
"""

import re
from dataclasses import dataclass, field

NOTE = "Some content was removed because it could not be verified."
EMPTY = "I couldn't produce an answer that is supported by the available sources."

# [doc §N] (canonical) or (doc §N), which small models sometimes write instead.
CITATION_RE = re.compile(r"[\[(]([^\[\]()§]+?)\s*§\s*(\d+)[^\])]*[\])]")
_ID_RE = re.compile(r"\b(?:[A-Z]{1,3}-?\d+|T\d{6}|C\d{4}|L\d{5})\b")  # C0042, T000123, RM-1, AML-01
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:[T ][\d:]+)?\b")
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], start=1)}
_TEXT_DATE_RE = re.compile(
    r"\b(?:(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})|(" + "|".join(_MONTHS)
    + r")\s+(\d{1,2}),?\s+(\d{4}))\b", re.I)
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])(?:\s+|$)", re.M)
_BARE_MARKER = re.compile(r"^\s*\d+[.)]\s*$")
# A currency the evidence never mentions is a hallucination too (e.g. "$" for EUR amounts).
_FOREIGN_CURRENCY = {"USD": re.compile(r"\$|\bUSD\b|\bdollars?\b", re.I),
                     "GBP": re.compile(r"£|\bGBP\b"), "CHF": re.compile(r"\bCHF\b")}
_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:[€$£]\s?|EUR\s?)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?"
    r"(?:\s?(%|k\b|K\b|m\b|M\b|mn\b|million\b|bn\b|billion\b|thousand\b))?", re.I)
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z(*•-])")


@dataclass
class Num:
    text: str
    value: float
    decimals: int
    mult: float


@dataclass
class VerifyResult:
    answer: str
    verified: bool
    removed: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    bad_citations: list[str] = field(default_factory=list)


def _slug(s: str) -> str:
    s = re.sub(r"\.md$", "", s.strip().lower())
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s.removesuffix("_policy")


def _dates_to_iso(text: str) -> tuple[str, list[str]]:
    """Replace dates with a placeholder, returning ISO strings found."""
    found = []

    def iso(m):
        found.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        return " "

    def txt(m):
        if m.group(1):
            d, mon, y = m.group(1), m.group(2), m.group(3)
        else:
            mon, d, y = m.group(4), m.group(5), m.group(6)
        found.append(f"{y}-{_MONTHS[mon.lower()]:02d}-{int(d):02d}")
        return " "

    text = _ISO_DATE_RE.sub(iso, text)
    text = _TEXT_DATE_RE.sub(txt, text)
    return text, found


def extract_numbers(text: str) -> tuple[list[Num], list[str]]:
    """Numbers (with display precision and unit multiplier) and ISO dates in `text`."""
    text = CITATION_RE.sub(" ", text)
    text, dates = _dates_to_iso(text)
    text = _ID_RE.sub(" ", text)
    text = _LIST_MARKER_RE.sub(" ", text)
    nums = []
    for m in _NUMBER_RE.finditer(text):
        whole, frac, unit = m.group(1), m.group(2) or "", (m.group(3) or "").lower()
        value = float(whole.replace(",", "") + frac)
        mult = _MULT.get(unit, 1.0)
        nums.append(Num(m.group(0).strip(), value * mult, max(len(frac) - 1, 0), mult))
    return nums, dates


def _supported(n: Num, evidence: list[float]) -> bool:
    for e in evidence:
        if abs(e - n.value) < 1e-6:
            return True
        # The answer may show an evidence value at lower precision: 954342.98 -> "954,343" / "0.95M".
        if round(e / n.mult, n.decimals) == round(n.value / n.mult, n.decimals) and e != 0:
            return True
    return False


def _match_citation(doc: str, section: str, chunks: dict[tuple[str, str], str]) -> tuple[str, str] | None:
    """Map a written citation to a retrieved chunk key, tolerating 'Overdraft Fee Waiver Policy', '.md'."""
    cited = _slug(doc)
    for a_doc, a_sec in chunks:
        if a_sec == section and (cited == a_doc or set(a_doc.split("_")) <= set(cited.split("_"))):
            return (a_doc, a_sec)
    return None


_CITATION_ONLY = re.compile(r"^\s*(?:\[[^\[\]]*§[^\[\]]*\]\s*)+[.;,]?\s*$")


def split_sentences(answer: str) -> list[str]:
    """Split into sentences/lines. A trailing citation ("... period. [doc §2]") stays with its sentence."""
    parts: list[str] = []
    for line in answer.splitlines():
        if not line.strip():
            parts.append("")
            continue
        for s in _SENTENCE_SPLIT.split(line):
            if not s.strip():
                continue
            if _CITATION_ONLY.match(s) and parts and parts[-1]:
                parts[-1] = f"{parts[-1]} {s.strip()}"
            elif parts and _BARE_MARKER.match(parts[-1]):  # "1." split off a list item
                parts[-1] = f"{parts[-1]} {s.strip()}"
            else:
                parts.append(s)
    return parts


def _pool(texts: list[str]) -> tuple[list[float], set[str]]:
    values, dates = [], set()
    for t in texts:
        nums, ds = extract_numbers(t)
        values.extend(n.value for n in nums)
        dates.update(ds)
    return values, dates


def verify(answer: str, evidence_texts: list[str], chunks: dict[tuple[str, str], str],
           question: str = "") -> VerifyResult:
    """evidence_texts: tool results; chunks: {(doc, section_no): content} of retrieved policy text.

    A sentence WITHOUT citations may use any evidence. A sentence WITH citations must be backed
    by the chunks it cites (plus tool results and the question): a correct number with a wrong
    citation is a mis-attribution and is removed.
    """
    base = [*evidence_texts, question]
    all_nums, all_dates = _pool(base + list(chunks.values()))
    all_text = " ".join(base + list(chunks.values()))
    foreign = [c for c, rx in _FOREIGN_CURRENCY.items() if c not in all_text and not rx.search(all_text)]

    kept, removed, bad_nums, bad_cits = [], [], [], []
    for sent in split_sentences(answer):
        problems = False
        cited = []
        for m in CITATION_RE.finditer(sent):
            key = _match_citation(m.group(1), m.group(2), chunks)
            if key is None:
                bad_cits.append(m.group(0))
                problems = True
            else:
                cited.append(chunks[key])
        ev_nums, ev_dates = _pool(base + cited) if cited else (all_nums, all_dates)
        nums, dates = extract_numbers(sent)
        for n in nums:
            if not _supported(n, ev_nums):
                bad_nums.append(n.text)
                problems = True
        for d in dates:
            if d not in ev_dates:
                bad_nums.append(d)
                problems = True
        for cur in foreign:
            if m := _FOREIGN_CURRENCY[cur].search(sent):
                bad_nums.append(m.group(0))
                problems = True
        (removed if problems else kept).append(sent)

    if not removed:
        return VerifyResult(answer=answer, verified=True)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    text = f"{text}\n\n_{NOTE}_" if text else f"{EMPTY}\n\n_{NOTE}_"
    return VerifyResult(answer=text, verified=False, removed=removed,
                        unsupported_numbers=bad_nums, bad_citations=bad_cits)
