"""Candidate-term retrieval over the coded-term vocabulary (char n-gram TF-IDF + fuzzy rerank)."""
from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from ..text import norm


class TermIndex:
    def __init__(self, terms: list[str]):
        self.terms = list(terms)
        self.norm_terms = [norm(t) for t in self.terms]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True)
        self.M = self.vec.fit_transform(self.norm_terms)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        nq = norm(query)
        sims = (self.M @ self.vec.transform([nq]).T).toarray().ravel()
        top = np.argsort(-sims, kind="stable")[: k * 3]
        scored = [(self.terms[i], 0.6 * float(sims[i]) + 0.4 * fuzz.token_set_ratio(nq, self.norm_terms[i]) / 100.0)
                  for i in top]
        scored.sort(key=lambda x: (-x[1], x[0]))      # canonical tie-break: reruns must build identical prompts
        return scored[:k]

    def candidates(self, queries: list[str], k: int, n_prior: int = 0, per_query: int = 4) -> list[str]:
        """Retrieved terms for the narrative clauses, plus the n_prior most frequent vocabulary terms.

        self.terms must be ordered by frequency (term_vocabulary() returns them that way). Coders use common
        terms (Pyrexia, Headache, ...) even when the narrative words differ, so a prior raises candidate recall.
        """
        out = self.search_many(queries, k=k, per_query=per_query)
        for t in self.terms[:n_prior]:
            if t not in out:
                out.append(t)
        return out

    def search_many(self, queries: list[str], k: int, per_query: int = 8) -> list[str]:
        """Interleave each query's top candidates so every extracted event gets representation."""
        lists = [[t for t, _ in self.search(q, per_query)] for q in queries if q.strip()]
        out: list[str] = []
        for rank in range(per_query):
            for lst in lists:
                if rank < len(lst) and lst[rank] not in out:
                    out.append(lst[rank])
        return out[:k]
