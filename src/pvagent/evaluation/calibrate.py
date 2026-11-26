"""Post-hoc confidence calibration: fit on a DEV run, evaluate on a TEST run (never fit on test).

'Correct' for routing = the case's seriousness agrees with the VAERS flags AND event-coding case-F1 >= 0.5.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
from sklearn.linear_model import LogisticRegression

from ..config import load_config, path
from ..data.sample import get_cases
from ..text import norm
from . import metrics as M

FEATURES = ["self_rating", "span_pass", "triage_agree", "n_events", "frac_coded", "sqrt_narrative_len",
            "n_rejected", "lr_score", "patient_identified"]
SETS = {"three_confidence_parts": [0, 1, 2], "all_features": list(range(len(FEATURES)))}


def _load(run: Path, con):
    cs = [json.loads(l) for l in open(Path(run) / "cases.jsonl", encoding="utf-8") if l.strip()]
    cs = [c for c in cs if not c.get("error")]
    g = get_cases(con, ids=[c["case_id"] for c in cs]).set_index("VAERS_ID")
    X, y = [], []
    for c in cs:
        p, ev, s = c["confidence_parts"], c["events"], c["seriousness"]
        X.append([p.get("self", 0), p.get("span", 1), p.get("triage_agree", 1), len(ev),
                  sum(e["coded_term"] is not None for e in ev) / max(1, len(ev)), len(c["narrative"]) ** 0.5,
                  len(c["rejected_fields"]), float(s.get("lr_score") or 0), float(c["validity"]["identifiable_patient"])])
        pred = {norm(e["coded_term"]) for e in ev if e["coded_term"]}
        gold = {norm(t) for t in g.loc[c["case_id"], "terms"]}
        ok = bool(s["serious"]) == bool(g.loc[c["case_id"], "SERIOUS"]) and M.set_prf([pred], [gold])["case_f1_mean"] >= 0.5
        y.append(int(ok))
    return np.array(X, float), np.array(y), np.array([c["confidence"] for c in cs])


def run_calibration(dev_run: Path, test_run: Path) -> dict:
    con = duckdb.connect(str(path(load_config()["db_path"])), read_only=True)
    Xd, yd, _ = _load(dev_run, con)
    Xt, yt, ct = _load(test_run, con)
    con.close()

    def summary(conf):
        rc = M.risk_coverage(conf, yt)
        at = lambda cov: min(rc, key=lambda r: abs(r["coverage"] - cov))["risk"]
        return {"ece": M.ece(conf, yt)["ece"], "auroc": M.auroc(conf, yt), "error_at_50pct_coverage": at(0.5),
                "error_at_80pct_coverage": at(0.8)}
    out = {"dev_run": Path(dev_run).name, "test_run": Path(test_run).name, "n_dev": len(yd), "n_test": len(yt),
           "case_error_rate_all": float(1 - yt.mean()), "hand_weighted": summary(ct)}
    for name, cols in SETS.items():
        mu, sd = Xd[:, cols].mean(0), Xd[:, cols].std(0) + 1e-9
        model = LogisticRegression(C=0.5, max_iter=2000).fit((Xd[:, cols] - mu) / sd, yd)
        out[name] = summary(model.predict_proba((Xt[:, cols] - mu) / sd)[:, 1])
    out["univariate_auroc_test"] = {n: M.auroc(Xt[:, i], yt) for i, n in enumerate(FEATURES)}
    (Path(test_run) / "calibration_posthoc.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
