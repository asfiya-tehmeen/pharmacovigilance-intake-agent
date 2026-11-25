"""Deterministic evidence-span verifier: a field survives only if its quote appears in the narrative."""
from __future__ import annotations

import re

_ws = re.compile(r"\s+")
_punct = re.compile(r"[\"'`“”‘’]")


def _canon(s: str) -> str:
    return _ws.sub(" ", _punct.sub("", (s or "").lower())).strip()


def quote_in_text(quote: str, text: str, min_len: int = 2) -> bool:
    """Case/whitespace/quote-mark-insensitive exact substring check. No fuzzy matching on purpose."""
    q = _canon(quote)
    if len(q) < min_len:
        return False
    return q in _canon(text)


def verify_case(case, narrative: str) -> list[str]:
    """Mark every Evidence on the case; drop events/products whose quote fails. Returns rejected field names."""
    rejected: list[str] = []

    def check(ev, name):
        if not ev.quote:
            ev.verified = None
            return True
        ev.verified = quote_in_text(ev.quote, narrative)
        if not ev.verified:
            rejected.append(name)
        return ev.verified

    check(case.patient.evidence, "patient")
    check(case.reporter.evidence, "reporter")
    kept_products = []
    for i, p in enumerate(case.products):
        # a product with no quote is kept only if its name is literally in the text
        ok = check(p.evidence, f"product[{i}]") if p.evidence.quote else quote_in_text(p.name, narrative)
        if ok:
            kept_products.append(p)
        elif not p.evidence.quote:
            rejected.append(f"product[{i}]")
    case.products = kept_products
    kept_events = []
    for i, e in enumerate(case.events):
        q = e.evidence.quote or e.verbatim
        e.evidence.quote = q
        if check(e.evidence, f"event[{i}]"):
            kept_events.append(e)
    case.events = kept_events
    for i, ev in enumerate(case.seriousness.evidence):
        check(ev, f"seriousness_evidence[{i}]")
    return rejected
