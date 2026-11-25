"""Disproportionality (PRR / ROR) on (product, event) pairs and comparison of two codings.

Signal rule (Evans et al. 2001 style): a >= min_count, PRR >= prr_threshold, chi-square >= chi2_threshold.
Counts are reports, not people; VAERS cannot establish causality or incidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr  # sklearn depends on scipy, so it is always present


def disproportionality(pairs: pd.DataFrame, min_count: int = 3, prr_thr: float = 2.0, chi2_thr: float = 4.0) -> pd.DataFrame:
    """pairs: columns report_id, product, term (one row per distinct triple)."""
    pairs = pairs.drop_duplicates(["report_id", "product", "term"])
    N = pairs["report_id"].nunique()
    per_prod = pairs.groupby("product")["report_id"].nunique()
    per_term = pairs.groupby("term")["report_id"].nunique()
    a = pairs.groupby(["product", "term"])["report_id"].nunique().rename("a").reset_index()
    a["n_prod"] = a["product"].map(per_prod)
    a["n_term"] = a["term"].map(per_term)
    a["b"] = a["n_prod"] - a["a"]
    a["c"] = a["n_term"] - a["a"]
    a["d"] = N - a["a"] - a["b"] - a["c"]
    A, B, C, D = (a[k].astype(float) for k in "abcd")
    with np.errstate(divide="ignore", invalid="ignore"):
        a["prr"] = (A / (A + B)) / (C / (C + D))
        se_ln_prr = np.sqrt(1 / A - 1 / (A + B) + 1 / C - 1 / (C + D))
        a["prr_lo"] = np.exp(np.log(a["prr"]) - 1.96 * se_ln_prr)
        a["ror"] = (A * D) / (B * C)
        se_ln_ror = np.sqrt(1 / A + 1 / B + 1 / C + 1 / D)
        a["ror_lo"] = np.exp(np.log(a["ror"]) - 1.96 * se_ln_ror)
        exp_a = (A + B) * (A + C) / N
        # Yates-corrected chi-square
        chi = N * (np.abs(A * D - B * C) - N / 2) ** 2 / ((A + B) * (C + D) * (A + C) * (B + D))
        a["chi2"] = chi
        a["expected_a"] = exp_a
    a["signal"] = (a["a"] >= min_count) & (a["prr"] >= prr_thr) & (a["chi2"] >= chi2_thr)
    return a.replace([np.inf, -np.inf], np.nan).sort_values(["signal", "prr_lo"], ascending=[False, False])


def compare(ref: pd.DataFrame, alt: pd.DataFrame, top_n: int = 50) -> dict:
    key = ["product", "term"]
    rs = set(map(tuple, ref.loc[ref["signal"], key].to_numpy()))
    as_ = set(map(tuple, alt.loc[alt["signal"], key].to_numpy()))
    m = ref[key + ["prr"]].merge(alt[key + ["prr"]], on=key, how="inner", suffixes=("_ref", "_alt")).dropna()
    rho = float(spearmanr(m["prr_ref"], m["prr_alt"]).statistic) if len(m) > 2 else None

    def top(df):
        s = df[df["signal"]].sort_values("prr_lo", ascending=False).head(top_n)
        return list(map(tuple, s[key].to_numpy()))
    tr, ta = top(ref), top(alt)
    return {
        "ref_signals": len(rs), "alt_signals": len(as_), "kept": len(rs & as_),
        "lost": len(rs - as_), "gained": len(as_ - rs),
        "jaccard": len(rs & as_) / len(rs | as_) if rs | as_ else 1.0,
        "spearman_prr_shared_pairs": rho, "shared_pairs": len(m),
        f"top{top_n}_overlap": len(set(tr) & set(ta)), f"top{top_n}_changed": top_n - len(set(tr) & set(ta)) if tr else None,
        "lost_examples": sorted(rs - as_)[:15], "gained_examples": sorted(as_ - rs)[:15],
    }


def simulate_coding(official: pd.DataFrame, term_recall: dict[str, float], default_recall: float,
                    fp_per_case: float, fp_terms: list[str], fp_weights: list[float] | None, seed: int = 0,
                    recall_scale: float = 1.0, fp_scale: float = 1.0) -> pd.DataFrame:
    """Apply an empirical error model (measured on the LLM eval sample) to the full official coding.

    Each official (report, term) is kept with probability recall(term); each report gains Poisson(fp_per_case)
    spurious terms drawn from the observed false-positive distribution. Products are left as officially coded,
    so the comparison isolates event-coding error.
    """
    rng = np.random.default_rng(seed)
    o = official.drop_duplicates(["report_id", "product", "term"])
    rt = o[["report_id", "term"]].drop_duplicates()
    rec = rt["term"].map(lambda t: term_recall.get(t.lower(), default_recall)).to_numpy() * recall_scale
    kept_rt = rt[rng.random(len(rt)) < np.clip(rec, 0, 1)]
    reports = rt["report_id"].unique()
    n_fp = rng.poisson(max(0.0, fp_per_case * fp_scale), size=len(reports))
    w = None
    if fp_weights is not None and len(fp_weights):
        w = np.asarray(fp_weights, float)
        w = w / w.sum()
    add = []
    if fp_terms:
        for rid, k in zip(reports, n_fp):
            if k:
                for t in rng.choice(fp_terms, size=k, p=w):
                    add.append((rid, t))
    new_rt = pd.concat([kept_rt, pd.DataFrame(add, columns=["report_id", "term"])], ignore_index=True).drop_duplicates()
    prods = o[["report_id", "product"]].drop_duplicates()
    return new_rt.merge(prods, on="report_id")[["report_id", "product", "term"]]
