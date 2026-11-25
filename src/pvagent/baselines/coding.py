"""Event-coding baselines: exact dictionary match and TF-IDF nearest term per clause.

Both are cheap enough to run on every report, which is what the full-data signal comparison needs.
"""
from __future__ import annotations

import re

import numpy as np

from ..text import clauses
from .retrieval import TermIndex

_tok = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return _tok.findall((s or "").lower())


class DictionaryCoder:
    """Predict every vocabulary term whose exact (normalised) wording appears in the narrative."""

    def __init__(self, terms: list[str], max_len: int = 6):
        self.lookup: dict[str, str] = {}
        for t in terms:
            k = " ".join(_tokens(t))
            if len(k) >= 3 and len(k.split()) <= max_len:
                self.lookup.setdefault(k, t)
        self.max_len = max_len

    def code(self, text: str) -> list[str]:
        toks = _tokens(text)
        out = set()
        for n in range(1, self.max_len + 1):
            for i in range(len(toks) - n + 1):
                hit = self.lookup.get(" ".join(toks[i:i + n]))
                if hit:
                    out.add(hit)
        return sorted(out)


class NearestTermCoder:
    """For each clause, take the nearest vocabulary term (char n-gram cosine) if it clears a threshold."""

    def __init__(self, index: TermIndex, threshold: float = 0.6, max_clauses: int = 60):
        self.index, self.threshold, self.max_clauses = index, threshold, max_clauses

    def code(self, text: str) -> list[str]:
        return self.code_many([text])[0]

    def code_many(self, texts: list[str], batch: int = 20000) -> list[list[str]]:
        owners, cl = [], []
        for i, t in enumerate(texts):
            cs = clauses(t)[: self.max_clauses]
            owners += [i] * len(cs)
            cl += [c.lower() for c in cs]
        out: list[set] = [set() for _ in texts]
        for s in range(0, len(cl), batch):
            Q = self.index.vec.transform(cl[s:s + batch])
            S = (Q @ self.index.M.T).tocsr()
            best = np.asarray(S.argmax(axis=1)).ravel()
            val = np.asarray(S.max(axis=1).todense()).ravel()
            for j, (b, v) in enumerate(zip(best, val)):
                if v >= self.threshold:
                    out[owners[s + j]].add(self.index.terms[b])
        return [sorted(x) for x in out]
