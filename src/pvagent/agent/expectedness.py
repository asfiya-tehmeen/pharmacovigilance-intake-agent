"""Expectedness: is a coded event mentioned in the product label's Adverse Reactions section?

Deterministic (no LLM): normalised term match, then a strict fuzzy match against label sentences.
This is a screening heuristic, not a regulatory expectedness assessment.
"""
from __future__ import annotations

import re

import duckdb
from rapidfuzz import fuzz

from ..text import norm


class ExpectednessChecker:
    def __init__(self, con: duckdb.DuckDBPyConnection, fuzzy_threshold: int = 92):
        self.by_type: dict[str, list[tuple[str, list[str]]]] = {}
        self.threshold = fuzzy_threshold
        try:
            rows = con.execute("""
                SELECT DISTINCT v.VAX_TYPE, l.vax_name, l.ar_text FROM labels l
                JOIN vax v ON v.VAX_NAME = l.vax_name WHERE length(l.ar_text) > 0
            """).fetchall()
        except duckdb.CatalogException:
            rows = []
        for vt, name, text in rows:
            sents = [norm(s) for s in re.split(r"(?<=[.;])\s+|\n", text) if len(s) > 3]
            self.by_type.setdefault(vt, []).append((name, sents))

    @property
    def available(self) -> bool:
        return bool(self.by_type)

    def check(self, term: str, vax_types: list[str]) -> tuple[str, str | None]:
        labels = [lab for vt in vax_types for lab in self.by_type.get(vt, [])]
        if not labels:
            return "no_label", None
        t = norm(term)
        pat = re.compile(r"\b" + re.escape(t) + r"\b")
        for name, sents in labels:
            for s in sents:
                if pat.search(s):
                    return "listed", name
        for name, sents in labels:
            for s in sents:
                if len(t) >= 6 and fuzz.partial_ratio(t, s) >= self.threshold:
                    return "listed", name
        return "unlisted", None
