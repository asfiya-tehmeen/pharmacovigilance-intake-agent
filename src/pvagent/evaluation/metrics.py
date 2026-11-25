"""Metrics for triage, coding, product extraction and calibration."""
from __future__ import annotations

import numpy as np

from ..text import norm


# ---------- set-valued (event coding, products) ----------
def set_prf(pred: list[set], gold: list[set]) -> dict:
    """Micro P/R/F1 over all (case, term) pairs, mean per-case F1 (macro over cases), exact-match rate."""
    tp = fp = fn = 0
    per_case, exact = [], 0
    for p, g in zip(pred, gold):
        p = {norm(x) for x in p}
        g = {norm(x) for x in g}
        t = len(p & g)
        tp, fp, fn = tp + t, fp + len(p - g), fn + len(g - p)
        prec = t / len(p) if p else (1.0 if not g else 0.0)
        rec = t / len(g) if g else 1.0
        per_case.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        exact += int(p == g)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"micro_precision": P, "micro_recall": R, "micro_f1": 0.0 if P + R == 0 else 2 * P * R / (P + R),
            "case_f1_mean": float(np.mean(per_case)) if per_case else 0.0,
            "exact_match": exact / len(pred) if pred else 0.0, "n": len(pred)}


def per_term_counts(pred: list[set], gold: list[set]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for p, g in zip(pred, gold):
        p = {norm(x) for x in p}
        g = {norm(x) for x in g}
        for t in p | g:
            d = out.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
            d["tp"] += int(t in p and t in g)
            d["fp"] += int(t in p and t not in g)
            d["fn"] += int(t in g and t not in p)
    return out


# ---------- triage ----------
def binary(y_true, y_pred) -> dict:
    y, p = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {"recall": rec, "precision": prec, "f1": 0.0 if rec + prec == 0 else 2 * rec * prec / (rec + prec),
            "specificity": tn / (tn + fp) if tn + fp else 0.0, "flagged_rate": float(p.mean()) if len(p) else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(y)}


def workload_at_recall(y_true, score, target: float = 0.99) -> dict:
    """Smallest share of cases a human must review (highest scores first) to reach target recall."""
    y, s = np.asarray(y_true, bool), np.asarray(score, float)
    if y.sum() == 0:
        return {"target_recall": target, "workload": None}
    order = np.argsort(-s, kind="stable")
    cum = np.cumsum(y[order]) / y.sum()
    k = int(np.searchsorted(cum, target) + 1)
    return {"target_recall": target, "workload": k / len(y), "threshold": float(s[order][k - 1]),
            "achieved_recall": float(cum[k - 1])}


def prevalence_adjusted_workload(flag_rate_pos: float, flag_rate_neg: float, prevalence: float) -> float:
    """Review workload at the natural serious-case rate (the eval sample oversamples serious cases)."""
    return flag_rate_pos * prevalence + flag_rate_neg * (1 - prevalence)


# ---------- calibration ----------
def ece(conf, correct, n_bins: int = 10) -> dict:
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(c, bins[1:-1]), 0, n_bins - 1)
    rows, e = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        acc, cf = float(y[m].mean()), float(c[m].mean())
        e += m.mean() * abs(acc - cf)
        rows.append({"bin": b, "n": int(m.sum()), "confidence": cf, "accuracy": acc})
    return {"ece": float(e), "reliability": rows}


def risk_coverage(conf, correct) -> list[dict]:
    """Error rate among auto-processed cases as coverage (share auto-processed) varies."""
    c, y = np.asarray(conf, float), np.asarray(correct, bool)
    order = np.argsort(-c, kind="stable")
    out = []
    for k in range(1, len(c) + 1):
        if k % max(1, len(c) // 50) and k != len(c):
            continue
        sel = y[order[:k]]
        out.append({"coverage": k / len(c), "risk": float(1 - sel.mean()), "threshold": float(c[order[k - 1]])})
    return out


def auroc(score, label) -> float | None:
    s, y = np.asarray(score, float), np.asarray(label, bool)
    if y.all() or (~y).all():
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))
