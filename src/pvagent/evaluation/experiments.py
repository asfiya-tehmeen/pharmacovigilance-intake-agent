"""Experiments from ideation.md section 6. Every run writes results/<run_id>/ (cases.jsonl + metrics.json)."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ..agent.expectedness import ExpectednessChecker
from ..agent.intake import IntakeAgent, _candidates
from ..agent.llm import BudgetExceeded, LLMClient
from ..agent.schema import SafetyCase
from ..baselines.coding import DictionaryCoder, NearestTermCoder
from ..baselines.product import ProductMatcher
from ..baselines.retrieval import TermIndex
from ..baselines.triage_lr import fit_predict, score_cases
from ..baselines.triage_rules import triage as rule_triage
from ..config import load_config, path
from ..data.sample import get_cases, product_vocabulary, term_vocabulary
from ..text import norm
from . import metrics as M
from .perturb import PERTURBATIONS
from .signal import compare, disproportionality, simulate_coding



def _con(read_only: bool = True):
    return duckdb.connect(str(path(load_config()["db_path"])), read_only=read_only)


def results_dir() -> Path:
    return path(load_config().get("results_dir", "results"))


def _run_dir(name: str) -> Path:
    d = results_dir() / f"{datetime.now():%Y%m%d-%H%M%S-%f}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dump(d: Path, obj: dict, fname: str = "metrics.json"):
    (d / fname).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _gold_sets(df: pd.DataFrame, col: str) -> list[set]:
    return [set(x) if x is not None and len(x) else set() for x in df[col]]


def _family(vt: str) -> str:
    return re.sub(r"[0-9X]*(-\d+)?$", "", (vt or "").upper()) or (vt or "").upper()


def population_serious_rate(con) -> float:
    return float(con.execute("SELECT avg(SERIOUS::INT) FROM reports WHERE length(SYMPTOM_TEXT) >= 30").fetchone()[0])


# --------------------------------------------------------------------------- baselines
def run_baselines() -> Path:
    cfg = load_config()
    con = _con()
    dev, test = get_cases(con, "dev"), get_cases(con, "test")
    vocab = term_vocabulary(con, cfg["coding"]["min_term_freq"])
    terms = vocab["TERM"].tolist()
    index = TermIndex(terms)
    prev = population_serious_rate(con)
    out: dict = {"n_dev": len(dev), "n_test": len(test), "vocab_size": len(terms), "population_serious_rate": prev}
    preds = pd.DataFrame({"VAERS_ID": test["VAERS_ID"], "serious": test["SERIOUS"]})

    # triage
    y = test["SERIOUS"].to_numpy(bool)
    rr = [rule_triage(t) for t in test["SYMPTOM_TEXT"]]
    preds["rules_serious"] = [r.serious for r in rr]
    preds["rules_score"] = [r.score for r in rr]
    out["triage_rules"] = M.binary(y, preds["rules_serious"])
    p, thr = fit_predict(dev["SYMPTOM_TEXT"], dev["SERIOUS"], test["SYMPTOM_TEXT"], 0.99)
    preds["lr_prob"] = p
    out["triage_lr"] = {**M.binary(y, p >= thr), "threshold_dev_99recall": thr, "auroc": M.auroc(p, y),
                        "workload_at_99_recall_oracle": M.workload_at_recall(y, p, 0.99)}
    for name, flag in (("triage_rules", preds["rules_serious"].to_numpy()), ("triage_lr", p >= thr)):
        b = out[name]
        b["workload_at_population_prevalence"] = M.prevalence_adjusted_workload(
            flag[y].mean() if y.any() else 0, flag[~y].mean() if (~y).any() else 0, prev)

    # coding
    gold = _gold_sets(test, "terms")
    dc, nc = DictionaryCoder(terms), NearestTermCoder(index)
    pd_dict = [set(dc.code(t)) for t in test["SYMPTOM_TEXT"]]
    pd_near = [set(x) for x in nc.code_many(test["SYMPTOM_TEXT"].tolist())]
    out["coding_dictionary"] = M.set_prf(pd_dict, gold)
    out["coding_nearest"] = M.set_prf(pd_near, gold)
    out["gold_terms_in_vocab"] = float(np.mean([norm(t) in {norm(x) for x in terms} for g in gold for t in g]))
    cand = [set(_candidates(index, t, cfg["coding"]["n_candidates"], cfg["coding"]["n_prior"])) for t in test["SYMPTOM_TEXT"]]
    out["candidate_recall_ceiling"] = M.set_prf(cand, gold)["micro_recall"]
    preds["dict_terms"] = [sorted(x) for x in pd_dict]
    preds["near_terms"] = [sorted(x) for x in pd_near]

    # product
    pm = ProductMatcher(product_vocabulary(con))
    gold_p = _gold_sets(test, "vax_types")
    pp = [set(pm.match(t)) for t in test["SYMPTOM_TEXT"]]
    out["product_dictionary"] = M.set_prf(pp, gold_p)
    out["product_dictionary_family"] = M.set_prf([{_family(x) for x in s} for s in pp],
                                                 [{_family(x) for x in s} for s in gold_p])
    d = _run_dir("baselines")
    preds.to_json(d / "predictions.jsonl", orient="records", lines=True)
    _dump(d, out)
    con.close()
    return d


# --------------------------------------------------------------------------- agent
def stratified_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    frac = n / len(df)
    parts = [g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed) for _, g in df.groupby("SERIOUS")]
    return pd.concat(parts).sort_values("VAERS_ID")


_LR_CACHE: dict = {}


def lr_triage_scores(con):
    """LR seriousness scores for all sample cases (dev out-of-fold, test from the dev-fit model). Cached per process."""
    if "v" not in _LR_CACHE:
        dev, test = get_cases(con, "dev"), get_cases(con, "test")
        _LR_CACHE["v"] = score_cases(dev["VAERS_ID"], dev["SYMPTOM_TEXT"], dev["SERIOUS"],
                                     test["VAERS_ID"], test["SYMPTOM_TEXT"], 0.99)
    return _LR_CACHE["v"]


def lr_model_on_dev(con):
    """The logistic-regression triage model fit on all dev cases (same model that scores test cases). Cached."""
    if "m" not in _LR_CACHE:
        from ..baselines.triage_lr import make_model
        dev = get_cases(con, "dev")
        _LR_CACHE["m"] = make_model().fit(list(dev["SYMPTOM_TEXT"]), dev["SERIOUS"].astype(int).to_numpy())
    return _LR_CACHE["m"]


def make_agent(con, model_size: str = "small", mode: str = "retrieval", hybrid: bool = False,
               consistency_k: int | None = None, provider: str | None = None) -> IntakeAgent:
    cfg = load_config()
    lc = cfg["llm"]
    prov = provider or lc["provider"]
    name_for = {"groq": {"small": lc["small_model"], "large": lc["large_model"]},
                "gemini": {"small": lc["gemini_model"], "large": lc["gemini_model"]},
                "ollama": {"small": lc["ollama_model"], "large": lc["ollama_model"]}}[prov]
    mk = lambda size: LLMClient(prov, name_for[size], lc["daily_token_budget"], lc["max_tokens"])
    small = mk("large" if (model_size == "large" and not hybrid) else "small")
    large = mk("large") if hybrid else None
    terms = term_vocabulary(con, cfg["coding"]["min_term_freq"])["TERM"].tolist()
    lr_scores, lr_thr = lr_triage_scores(con)
    return IntakeAgent(small, TermIndex(terms), ExpectednessChecker(con), large=large, mode=mode,
                       n_candidates=cfg["coding"]["n_candidates"], n_prior=cfg["coding"]["n_prior"], lr_scores=lr_scores, lr_threshold=lr_thr, threshold=cfg["routing"]["confidence_threshold"],
                       consistency_k=lc["self_consistency_k"] if consistency_k is None else consistency_k,
                       temperature=lc["temperature"])


def run_agent(split: str = "test", limit: int | None = None, model_size: str = "small", mode: str = "retrieval",
              hybrid: bool = False, consistency_k: int | None = 0, provider: str | None = None,
              perturbation: str | None = None, name: str | None = None, ids: list[int] | None = None,
              agent: IntakeAgent | None = None) -> Path:
    cfg = load_config()
    con = _con()
    cases = get_cases(con, split, ids=ids)
    if ids is None:
        cases = stratified_subset(cases, limit or cfg["sample"]["llm_subset"], cfg["sample"]["seed"])
    agent = agent or make_agent(con, model_size, mode, hybrid, consistency_k, provider)
    if perturbation:
        # The regression layer must see the same perturbed text as the rules and the LLM; precomputed scores are
        # for the clean narratives.
        texts = [PERTURBATIONS[perturbation](t, seed=int(i)) for t, i in zip(cases["SYMPTOM_TEXT"], cases["VAERS_ID"])]
        probs = lr_model_on_dev(con).predict_proba(texts)[:, 1]
        agent.lr_scores = {**agent.lr_scores, **{int(i): float(p) for i, p in zip(cases["VAERS_ID"], probs)}}
    tag = name or f"agent_{'hybrid' if hybrid else model_size}_{mode}" + (f"_{perturbation}" if perturbation else "")
    d = _run_dir(tag)
    meta = {"split": split, "n_planned": len(cases), "model_size": model_size, "mode": mode, "hybrid": hybrid,
            "consistency_k": agent.consistency_k, "provider": agent.small.provider, "small_model": agent.small.model,
            "large_model": agent.large.model if agent.large else None, "perturbation": perturbation,
            "started": datetime.now().isoformat()}
    _dump(d, meta, "run.json")
    done, stopped = 0, None
    t0 = time.time()
    try:
        with open(d / "cases.jsonl", "w", encoding="utf-8") as f:
            for row in cases.itertuples(index=False):
                text = row.SYMPTOM_TEXT
                if perturbation:
                    text = PERTURBATIONS[perturbation](text, seed=int(row.VAERS_ID))
                try:
                    case = agent.process(
                        int(row.VAERS_ID), text,
                        vax_types_for_label=list(row.vax_types) if row.vax_types is not None else None,
                        form={"age": row.AGE_YRS, "sex": row.SEX,
                              "has_vaccine": row.vax_types is not None and len(row.vax_types) > 0})
                except BudgetExceeded as e:
                    stopped = str(e)
                    break
                f.write(case.model_dump_json() + chr(10))
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(cases)} cases  ({time.time() - t0:.0f}s, {agent.small.used_today()} tokens today)")
    except Exception as e:      # keep partial results usable; rerun resumes from the cache
        stopped = f"crashed: {type(e).__name__}: {e}"[:300]
    meta.update({"n_done": done, "stopped_early": stopped, "finished": datetime.now().isoformat()})
    _dump(d, meta, "run.json")
    con.close()
    if done:
        evaluate_run(d)
    else:                      # nothing processed (e.g. quota already spent): do not leave an empty run behind
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        print(f"no cases processed ({stopped}); nothing saved")
    return d


def load_run(d: Path) -> tuple[list[SafetyCase], dict]:
    cases = [SafetyCase.model_validate_json(l) for l in open(d / "cases.jsonl", encoding="utf-8") if l.strip()]
    return cases, json.loads((d / "run.json").read_text(encoding="utf-8"))


def evaluate_run(d: Path) -> dict:
    d = Path(d)
    cases, meta = load_run(d)
    con = _con()
    gold = get_cases(con, ids=[c.case_id for c in cases]).set_index("VAERS_ID")
    prev = population_serious_rate(con)
    con.close()
    cases = [c for c in cases if c.case_id in gold.index]
    g = gold.loc[[c.case_id for c in cases]]
    y = g["SERIOUS"].to_numpy(bool)
    ok = [c for c in cases if c.error is None]
    out: dict = {"run": d.name, **{k: meta.get(k) for k in ("model_size", "mode", "hybrid", "perturbation", "provider")},
                 "n_cases": len(cases), "n_llm_errors": len(cases) - len(ok)}

    llm_s = np.array([c.seriousness.serious for c in cases])
    rule_s = np.array([bool(c.seriousness.rule_serious) for c in cases])
    lr_s = np.array([bool(c.seriousness.lr_serious) for c in cases])
    either = llm_s | rule_s | lr_s
    out["triage_lr"] = M.binary(y, lr_s)
    out["triage_llm_or_rules_only"] = M.binary(y, llm_s | rule_s)
    out["triage_llm"] = M.binary(y, llm_s)
    out["triage_rules"] = M.binary(y, rule_s)
    out["triage_llm_or_rules_or_lr"] = M.binary(y, either)
    out["triage_llm_or_rules_or_lr"]["workload_at_population_prevalence"] = M.prevalence_adjusted_workload(
        either[y].mean() if y.any() else 0, either[~y].mean() if (~y).any() else 0, prev)

    pred_terms = [{e.coded_term for e in c.events if e.coded_term} for c in cases]
    gold_terms = _gold_sets(g, "terms")
    out["coding"] = M.set_prf(pred_terms, gold_terms)
    pv = [{p.vax_type for p in c.products if p.vax_type} for c in cases]
    gv = _gold_sets(g, "vax_types")
    out["product"] = M.set_prf(pv, gv)
    out["product_family"] = M.set_prf([{_family(x) for x in s} for s in pv], [{_family(x) for x in s} for s in gv])

    # calibration: a case is "correct" if seriousness matches AND case-level coding F1 >= 0.5
    per_f1 = [M.set_prf([p], [q])["case_f1_mean"] for p, q in zip(pred_terms, gold_terms)]
    correct = (llm_s == y) & (np.array(per_f1) >= 0.5)
    out["case_correct_rate"] = float(correct.mean()) if len(correct) else None
    cal = {}
    for part in ("self", "span", "triage_agree", "consistency"):
        vals = [c.confidence_parts.get(part) for c in cases]
        m = np.array([v is not None for v in vals])
        if m.sum() >= 10:
            conf = np.array([v for v in vals if v is not None], float)
            cal[part] = {"ece": M.ece(conf, correct[m])["ece"], "auroc": M.auroc(conf, correct[m]), "n": int(m.sum())}
    conf = np.array([c.confidence for c in cases])
    cal["combined"] = {**M.ece(conf, correct), "auroc": M.auroc(conf, correct)}
    cal["risk_coverage"] = M.risk_coverage(conf, correct)
    out["calibration"] = cal

    # routing / human workload
    human = np.array([c.route == "human_review" for c in cases])
    missed = y & ~either
    out["routing"] = {"human_review_rate": float(human.mean()), "auto_rate": float(1 - human.mean()),
                      "serious_missed_by_agent_and_rules": int(missed.sum()),
                      "auto_error_rate": float(1 - correct[~human].mean()) if (~human).any() else None,
                      "reasons": pd.Series([r for c in cases for r in c.route_reasons]).value_counts().to_dict()}
    out["span_verifier"] = {"cases_with_rejected_fields": int(sum(bool(c.rejected_fields) for c in cases)),
                            "rejected_fields_total": int(sum(len(c.rejected_fields) for c in cases))}
    out["validity"] = {"valid_rate": float(np.mean([c.validity.valid for c in ok])) if ok else None}
    exp = pd.Series([e.expectedness for c in cases for e in c.events]).value_counts().to_dict()
    out["expectedness_counts"] = exp
    pt = sum(c.provenance.prompt_tokens for c in cases)
    ct = sum(c.provenance.completion_tokens for c in cases)
    out["cost"] = {"prompt_tokens": pt, "completion_tokens": ct, "tokens_per_case": (pt + ct) / max(1, len(cases)),
                   "seconds_per_case": float(np.mean([c.provenance.seconds for c in cases])) if cases else None,
                   "escalated": int(sum(c.provenance.escalated for c in cases)),
                   "note": "multiply by your provider's paid $/token to get a paid-tier cost"}
    out["per_term"] = M.per_term_counts(pred_terms, gold_terms)
    _dump(d, out)
    return out


# --------------------------------------------------------------------------- threshold sweep
def threshold_sweep(d: Path, grid=None) -> pd.DataFrame:
    cases, _ = load_run(Path(d))
    con = _con()
    gold = get_cases(con, ids=[c.case_id for c in cases]).set_index("VAERS_ID")
    con.close()
    y = np.array([bool(gold.loc[c.case_id, "SERIOUS"]) for c in cases])
    gt = [set(gold.loc[c.case_id, "terms"]) for c in cases]
    pt = [{e.coded_term for e in c.events if e.coded_term} for c in cases]
    f1 = np.array([M.set_prf([p], [q])["case_f1_mean"] for p, q in zip(pt, gt)])
    llm_s = np.array([c.seriousness.serious for c in cases])
    rule_s = np.array([bool(c.seriousness.rule_serious) or bool(c.seriousness.lr_serious) for c in cases])
    conf = np.array([c.confidence for c in cases])
    hard = np.array([any(r in ("triage_disagreement", "invalid_icsr") for r in c.route_reasons) for c in cases])
    rows = []
    for thr in grid if grid is not None else np.round(np.arange(0.0, 1.01, 0.05), 2):
        human = (conf < thr) | hard
        auto = ~human
        rows.append({"threshold": thr, "human_review_rate": human.mean(),
                     "serious_auto_processed_as_nonserious": int((auto & y & ~(llm_s | rule_s)).sum()),
                     "serious_total": int(y.sum()),
                     "auto_case_f1": float(f1[auto].mean()) if auto.any() else None})
    df = pd.DataFrame(rows)
    df.to_csv(Path(d) / "threshold_sweep.csv", index=False)
    return df


# --------------------------------------------------------------------------- signal impact
def _official_pairs(con) -> pd.DataFrame:
    return con.execute("""
        SELECT s.VAERS_ID AS report_id, v.VAX_TYPE AS product, s.TERM AS term
        FROM symptoms s JOIN vax v USING (VAERS_ID) JOIN reports r USING (VAERS_ID)
        WHERE length(r.SYMPTOM_TEXT) >= 30 AND NOT r.POSSIBLE_DUP
    """).df()


def run_signal(agent_run: Path | None = None, with_nearest: bool = False, sweep: bool = True) -> Path:
    cfg = load_config()
    sc = cfg["signal"]
    con = _con()
    official = _official_pairs(con)
    dis = lambda p: disproportionality(p, sc["min_count"], sc["prr_threshold"], sc["chi2_threshold"])
    ref = dis(official)
    out: dict = {"reports": int(official["report_id"].nunique()), "official_signals": int(ref["signal"].sum())}
    d = _run_dir("signal")
    ref.to_csv(d / "signals_official.csv", index=False)

    texts = con.execute("""SELECT VAERS_ID, SYMPTOM_TEXT FROM reports
                           WHERE length(SYMPTOM_TEXT) >= 30 AND NOT POSSIBLE_DUP""").df()
    prods = official[["report_id", "product"]].drop_duplicates()
    terms = term_vocabulary(con, cfg["coding"]["min_term_freq"])["TERM"].tolist()

    def pairs_from(codes: list[list[str]]) -> pd.DataFrame:
        rt = pd.DataFrame([(rid, t) for rid, ts in zip(texts["VAERS_ID"], codes) for t in ts], columns=["report_id", "term"])
        return rt.merge(prods, on="report_id")

    dc = DictionaryCoder(terms)
    alt = dis(pairs_from([dc.code(t) for t in texts["SYMPTOM_TEXT"]]))
    out["dictionary_vs_official"] = compare(ref, alt)
    if with_nearest:
        nc = NearestTermCoder(TermIndex(terms))
        alt = dis(pairs_from(nc.code_many(texts["SYMPTOM_TEXT"].tolist())))
        out["nearest_vs_official"] = compare(ref, alt)

    if agent_run:
        cases, _ = load_run(Path(agent_run))
        ids = [c.case_id for c in cases if c.error is None]
        gold = get_cases(con, ids=ids).set_index("VAERS_ID")
        pt = [{e.coded_term for e in c.events if e.coded_term} for c in cases if c.error is None]
        gt = [set(gold.loc[i, "terms"]) for i in ids]
        counts = M.per_term_counts(pt, gt)
        tp = sum(v["tp"] for v in counts.values())
        fn = sum(v["fn"] for v in counts.values())
        default_recall = tp / max(1, tp + fn)
        # per-term recall with shrinkage toward the global recall (k=5 pseudo-observations)
        term_recall = {t: (v["tp"] + 5 * default_recall) / (v["tp"] + v["fn"] + 5) for t, v in counts.items()}
        fp_terms = [t for t, v in counts.items() if v["fp"]]
        fp_w = [counts[t]["fp"] for t in fp_terms]
        fp_per_case = sum(fp_w) / max(1, len(ids))
        # map normalised fp terms back to vocabulary spelling
        spell = {norm(t): t for t in terms}
        fp_terms = [spell.get(t, t) for t in fp_terms]
        out["agent_error_model"] = {"from_run": Path(agent_run).name, "n_cases": len(ids),
                                    "global_recall": default_recall, "fp_per_case": fp_per_case}
        sims = []
        for seed in range(5):
            sim = dis(simulate_coding(official, term_recall, default_recall, fp_per_case, fp_terms, fp_w, seed))
            sims.append(compare(ref, sim))
        out["agent_simulated_vs_official"] = {
            k: {"mean": float(np.mean([s[k] for s in sims])), "min": float(np.min([s[k] for s in sims])),
                "max": float(np.max([s[k] for s in sims]))}
            for k in ("alt_signals", "kept", "lost", "gained", "jaccard", "top50_overlap")}
        out["agent_simulated_vs_official"]["example_seed0"] = {k: sims[0][k] for k in ("lost_examples", "gained_examples")}
        # direct comparison on the agent-processed sample (small; noisy)
        ag_rt = pd.DataFrame([(c.case_id, e.coded_term) for c in cases if c.error is None for e in c.events
                              if e.coded_term], columns=["report_id", "term"]).drop_duplicates()
        sample_off = official[official["report_id"].isin(ids)]
        out["agent_direct_on_sample"] = compare(dis(sample_off), dis(ag_rt.merge(prods, on="report_id")))

    if sweep:
        rows = []
        for rs in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
            for fp in (0.0, 0.5, 1.0, 2.0):
                sim = dis(simulate_coding(official, {}, rs, fp, terms[:500],
                                          list(range(500, 0, -1)), seed=0))
                c = compare(ref, sim)
                rows.append({"recall": rs, "fp_per_case": fp, **{k: c[k] for k in
                             ("alt_signals", "kept", "lost", "gained", "jaccard", "top50_overlap")}})
        sw = pd.DataFrame(rows)
        sw.to_csv(d / "error_level_sweep.csv", index=False)
        out["error_level_sweep_file"] = "error_level_sweep.csv"
    _dump(d, out)
    con.close()
    return d


# --------------------------------------------------------------------------- robustness
def run_robustness(limit: int = 100, model_size: str = "small", provider: str | None = None,
                   with_agent: bool = True) -> Path:
    """Clean vs perturbed on the same cases, for the agent and the cheap baselines."""
    cfg = load_config()
    con = _con()
    cases = stratified_subset(get_cases(con, "test"), limit, cfg["sample"]["seed"] + 1)
    ids = cases["VAERS_ID"].astype(int).tolist()
    terms = term_vocabulary(con, cfg["coding"]["min_term_freq"])["TERM"].tolist()
    dc = DictionaryCoder(terms)
    y = cases["SERIOUS"].to_numpy(bool)
    gold = _gold_sets(cases, "terms")
    con.close()
    out: dict = {"n": len(ids), "baselines": {}, "agent_runs": {}}
    for pname, fn in [("clean", None), *PERTURBATIONS.items()]:
        texts = [t if fn is None else fn(t, seed=int(i)) for t, i in zip(cases["SYMPTOM_TEXT"], ids)]
        out["baselines"][pname] = {
            "rules_recall": M.binary(y, [rule_triage(t).serious for t in texts])["recall"],
            "dictionary_f1": M.set_prf([set(dc.code(t)) for t in texts], gold)["micro_f1"]}
    d = _run_dir("robustness")
    for pname in (["clean", *PERTURBATIONS] if with_agent else []):
        try:
            rd = run_agent("test", limit=None, model_size=model_size, provider=provider, ids=ids,
                           perturbation=None if pname == "clean" else pname, name=f"robust_{pname}")
            m = json.loads((rd / "metrics.json").read_text(encoding="utf-8")) if (rd / "metrics.json").exists() else {}
            out["agent_runs"][pname] = {"run": rd.name, "n": m.get("n_cases"),
                                        "coding_f1": m.get("coding", {}).get("micro_f1"),
                                        "triage_llm_recall": m.get("triage_llm", {}).get("recall"),
                                        "triage_combined_recall": m.get("triage_llm_or_rules_or_lr", {}).get("recall")}
        except BudgetExceeded as e:
            out["agent_runs"][pname] = {"stopped": str(e)}
            break
    _dump(d, out)
    return d
