"""pv command-line interface. Run `pv -h` for commands."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from .config import ROOT, load_config, path


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def _latest(prefix: str) -> Path:
    runs = sorted(path(load_config().get("results_dir", "results")).glob(f"*_{prefix}*"))
    if not runs:
        sys.exit(f"no results/*_{prefix}* run found")
    return runs[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pv", description="Pharmacovigilance case-intake agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("load", help="load VAERS CSVs/zips from data/raw into DuckDB")
    sub.add_parser("sample", help="build (or reuse) the frozen stratified dev/test sample")
    lab = sub.add_parser("labels", help="fetch DailyMed adverse-reaction sections for top products")
    lab.add_argument("--top", type=int, default=30)
    sub.add_parser("baselines", help="run rule/LR triage, dictionary/nearest coding, product matcher")
    ag = sub.add_parser("agent", help="run the LLM intake agent on a stratified subset")
    ag.add_argument("--split", default="test", choices=["dev", "test"])
    ag.add_argument("--limit", type=int)
    ag.add_argument("--model", default="small", choices=["small", "large"])
    ag.add_argument("--mode", default="retrieval", choices=["retrieval", "prompt_only"])
    ag.add_argument("--hybrid", action="store_true", help="small model first, escalate low-confidence to large")
    ag.add_argument("--consistency-k", type=int, default=0, help="extra samples for self-consistency confidence")
    ag.add_argument("--provider", choices=["groq", "gemini", "ollama"])
    ag.add_argument("--like", help="run dir: use the first --limit case ids of that run (paired ablations)")
    ag.add_argument("--perturb", choices=["typos", "abbreviations", "lay_language", "messy"])
    ev = sub.add_parser("evaluate", help="(re)compute metrics for a run directory")
    ev.add_argument("run", nargs="?")
    sw = sub.add_parser("sweep", help="routing threshold sweep for a run")
    sw.add_argument("run", nargs="?")
    sg = sub.add_parser("signal", help="PRR/ROR signal impact: official vs baseline vs agent error model")
    sg.add_argument("--agent-run")
    sg.add_argument("--with-nearest", action="store_true")
    sg.add_argument("--no-sweep", action="store_true")
    rb = sub.add_parser("robustness", help="clean vs perturbed narratives (agent + baselines)")
    rb.add_argument("--limit", type=int, default=100)
    rb.add_argument("--baselines-only", action="store_true", help="skip the LLM (free): rules + dictionary only")
    rb.add_argument("--model", default="small", choices=["small", "large"])
    rb.add_argument("--provider", choices=["groq", "gemini", "ollama"])
    cb = sub.add_parser("calibrate", help="fit a confidence calibrator on a dev run, evaluate on a test run")
    cb.add_argument("--dev", required=True)
    cb.add_argument("--test", required=True)
    sub.add_parser("budget", help="show today's token usage per model")
    sub.add_parser("check", help="check keys, data files and DB state")
    sub.add_parser("review", help="launch the Streamlit human-review queue")
    a = ap.parse_args(argv)
    cfg = load_config()

    if a.cmd == "load":
        from .data.load import load_vaers
        _print(load_vaers(path(cfg["raw_dir"]), cfg["years"], path(cfg["db_path"])))
    elif a.cmd == "sample":
        from .data.sample import build_sample
        s = cfg["sample"]
        _print(build_sample(path(cfg["db_path"]), s["eval_size"], s["serious_fraction"], s["test_fraction"], s["seed"],
                            path("data/processed/sample_split.csv")))
    elif a.cmd == "labels":
        from .data.labels import fetch_labels
        _print(fetch_labels(path(cfg["db_path"]), a.top))
    elif a.cmd == "baselines":
        from .evaluation.experiments import run_baselines
        d = run_baselines()
        print(d)
        _print(json.loads((d / "metrics.json").read_text()))
    elif a.cmd == "agent":
        from .evaluation.experiments import run_agent
        ids = None
        if a.like:
            ids = sorted(json.loads(l)["case_id"] for l in open(Path(a.like) / "cases.jsonl", encoding="utf-8") if l.strip())
            ids = ids[: a.limit] if a.limit else ids
        d = run_agent(a.split, a.limit, a.model, a.mode, a.hybrid, a.consistency_k, a.provider, a.perturb, ids=ids)
        print(d)
        if (d / "metrics.json").exists():
            m = json.loads((d / "metrics.json").read_text())
            m.pop("per_term", None)
            m.get("calibration", {}).pop("risk_coverage", None)
            _print(m)
    elif a.cmd == "evaluate":
        from .evaluation.experiments import evaluate_run
        m = evaluate_run(Path(a.run) if a.run else _latest("agent"))
        m.pop("per_term", None)
        _print(m)
    elif a.cmd == "sweep":
        from .evaluation.experiments import threshold_sweep
        print(threshold_sweep(Path(a.run) if a.run else _latest("agent")).to_string(index=False))
    elif a.cmd == "signal":
        from .evaluation.experiments import run_signal
        d = run_signal(Path(a.agent_run) if a.agent_run else None, a.with_nearest, not a.no_sweep)
        print(d)
        _print(json.loads((d / "metrics.json").read_text()))
    elif a.cmd == "robustness":
        from .evaluation.experiments import run_robustness
        d = run_robustness(a.limit, a.model, a.provider, with_agent=not a.baselines_only)
        _print(json.loads((d / "metrics.json").read_text()))
    elif a.cmd == "calibrate":
        from .evaluation.calibrate import run_calibration
        _print(run_calibration(Path(a.dev), Path(a.test)))
    elif a.cmd == "budget":
        import sqlite3
        p = path("data/cache/llm_cache.sqlite")
        if not p.exists():
            print("no LLM calls yet")
            return
        db = sqlite3.connect(str(p))
        rows = db.execute("SELECT day, provider, model, sum(prompt), sum(completion), count(*) FROM usage "
                          "GROUP BY 1,2,3 ORDER BY 1 DESC").fetchall()
        print(f"budget per model per day: {cfg['llm']['daily_token_budget']}  (today: {date.today()})")
        for r in rows:
            print(f"{r[0]}  {r[1]:7s} {r[2]:28s} prompt={r[3]:>8} completion={r[4]:>7} calls={r[5]}")
    elif a.cmd == "check":
        from .config import env
        raw = path(cfg["raw_dir"])
        print("GROQ_API_KEY:", "set" if env("GROQ_API_KEY") else "MISSING")
        print("GEMINI_API_KEY:", "set" if env("GEMINI_API_KEY") else "not set (optional)")
        for y in cfg["years"]:
            z = raw / f"{y}VAERSData.zip"
            csvs = all((raw / f"{y}{t}.csv").exists() for t in ("VAERSDATA", "VAERSSYMPTOMS", "VAERSVAX"))
            print(f"VAERS {y}:", "zip found" if z.exists() else ("csvs found" if csvs else f"MISSING ({z})"))
        db = path(cfg["db_path"])
        if db.exists():
            import duckdb
            con = duckdb.connect(str(db), read_only=True)
            tabs = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            for t in tabs:
                print(f"table {t}: {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]} rows")
        else:
            print("database: not built yet (run `pv load`)")
    elif a.cmd == "review":
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "review_app.py")])


if __name__ == "__main__":
    main()
