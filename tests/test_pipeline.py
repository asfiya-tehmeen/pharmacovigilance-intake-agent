"""End-to-end on the synthetic fixture with a fake LLM: load -> sample -> baselines -> agent -> eval -> signal."""
import json
from pathlib import Path

import duckdb

from conftest import FakeLLM


def test_load_and_sample(project):
    con = duckdb.connect(project["db_path"], read_only=True)
    assert con.execute("SELECT count(*) FROM reports").fetchone()[0] == 160
    assert con.execute("SELECT count(*) FROM sample").fetchone()[0] == 120
    n_test_ser = con.execute("SELECT count(*) FROM sample WHERE split='test' AND SERIOUS").fetchone()[0]
    assert n_test_ser > 0
    con.close()


def test_frozen_split_is_reused(project, tmp_path):
    from pvagent.data.sample import build_sample
    before = (tmp_path / "split.csv").read_text()
    info = build_sample(Path(project["db_path"]), 50, 0.2, 0.5, 7, tmp_path / "split.csv")
    assert "reused_frozen_split" in info and (tmp_path / "split.csv").read_text() == before


def test_baselines_run(project):
    from pvagent.evaluation.experiments import run_baselines
    m = json.loads((run_baselines() / "metrics.json").read_text())
    assert m["triage_rules"]["recall"] == 1.0
    assert m["coding_dictionary"]["micro_recall"] > 0.5
    assert 0 <= m["candidate_recall_ceiling"] <= 1


def test_agent_eval_sweep_signal(project):
    from pvagent.agent.expectedness import ExpectednessChecker
    from pvagent.agent.intake import IntakeAgent
    from pvagent.baselines.retrieval import TermIndex
    from pvagent.data.sample import term_vocabulary
    from pvagent.evaluation.experiments import run_agent, run_signal, threshold_sweep

    con = duckdb.connect(project["db_path"], read_only=True)
    idx = TermIndex(term_vocabulary(con, 1)["TERM"].tolist())
    agent = IntakeAgent(FakeLLM(), idx, ExpectednessChecker(con), n_candidates=25, threshold=0.8)
    con.close()
    d = run_agent("test", limit=20, agent=agent)
    m = json.loads((d / "metrics.json").read_text())
    assert m["n_cases"] == 20 and m["n_llm_errors"] == 0
    assert m["coding"]["micro_f1"] > 0.8                       # fake model is near-perfect
    assert m["span_verifier"]["rejected_fields_total"] >= 20     # the planted fake quote is always dropped
    assert m["triage_llm_or_rules_or_lr"]["recall"] == 1.0
    sw = threshold_sweep(d)
    assert sw["human_review_rate"].is_monotonic_increasing
    s = json.loads((run_signal(agent_run=d, sweep=False) / "metrics.json").read_text())
    assert s["official_signals"] >= 1
    assert "agent_simulated_vs_official" in s and "dictionary_vs_official" in s


def test_vocabulary_order_is_canonical(project):
    """Prompts (and so LLM cache keys) depend on vocabulary order; ties must be broken by name, never arbitrarily."""
    from pvagent.data.sample import term_vocabulary
    con = duckdb.connect(project["db_path"], read_only=True)
    v = term_vocabulary(con, 1)
    assert v["n"].is_monotonic_decreasing
    for _, g in v.groupby("n"):
        assert list(g["TERM"]) == sorted(g["TERM"])
    con.close()
