import numpy as np
import pandas as pd

from pvagent.agent.intake import parse_case
from pvagent.agent.llm import parse_json
from pvagent.agent.verifier import quote_in_text, verify_case
from pvagent.baselines.coding import DictionaryCoder, NearestTermCoder
from pvagent.baselines.retrieval import TermIndex
from pvagent.baselines.triage_lr import threshold_for_recall
from pvagent.baselines.triage_rules import triage
from pvagent.evaluation import metrics as M
from pvagent.evaluation.perturb import PERTURBATIONS
from pvagent.evaluation.signal import compare, disproportionality, simulate_coding


def test_quote_verifier_exact_only():
    t = 'Pt developed a  "severe" Headache and fever.'
    assert quote_in_text("severe headache", t)
    assert quote_in_text("DEVELOPED A SEVERE", t)
    assert not quote_in_text("severe migraine", t)
    assert not quote_in_text("a", t)


def test_verifier_drops_unsupported_events():
    narr = "Patient had fever and rash after the flu shot."
    case = parse_case(1, narr, {"events": [{"verbatim": "fever", "quote": "fever", "term_ids": [0]},
                                           {"verbatim": "seizure", "quote": "had a seizure", "term_ids": [1]}],
                                "products": [{"name": "flu shot", "quote": "the flu shot"}]},
                      ["Pyrexia", "Seizure"])
    rejected = verify_case(case, narr)
    assert [e.coded_term for e in case.events] == ["Pyrexia"]
    assert "event[1]" in rejected and len(case.products) == 1


def test_parse_case_rejects_out_of_list_ids():
    case = parse_case(1, "fever", {"events": [{"verbatim": "fever", "term_ids": [0, 7]}]}, ["Pyrexia"])
    assert [e.coded_term for e in case.events] == ["Pyrexia"]
    assert any(f.startswith("invalid_term_ids") for f in case.rejected_fields)


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure: {"a": 2} done') == {"a": 2}


def test_rules_negation_and_er_not_serious():
    assert triage("Patient died at home.").serious
    r = triage("She was not hospitalized.")
    assert r.score == 0.5 and r.criteria == {}
    assert not triage("Seen in the ER, sent home with Tylenol.").serious
    assert "life_threatening" in triage("Developed anaphylaxis, given EpiPen").criteria


def test_dictionary_and_nearest_coders():
    terms = ["Headache", "Injection site pain", "Pyrexia", "Rash"]
    dc = DictionaryCoder(terms)
    assert dc.code("Severe headache; injection-site pain and RASH.") == ["Headache", "Injection site pain", "Rash"]
    nc = NearestTermCoder(TermIndex(terms), threshold=0.5)
    assert "Headache" in nc.code("bad headaches all day")


def test_set_prf_and_binary():
    m = M.set_prf([{"a", "b"}, {"c"}], [{"a"}, {"c", "d"}])
    assert np.isclose(m["micro_precision"], 2 / 3) and np.isclose(m["micro_recall"], 2 / 3)
    b = M.binary([1, 1, 0, 0], [1, 0, 1, 0])
    assert b["recall"] == 0.5 and b["precision"] == 0.5


def test_workload_and_threshold():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .6])
    assert M.workload_at_recall(y, s, 1.0)["workload"] == 0.4
    assert threshold_for_recall(y, s, 1.0) == 0.6


def test_ece_perfect_calibration():
    assert M.ece([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0])["ece"] == 0.0


def test_prr_matches_hand_calculation():
    # product X: 10 reports, 6 with term T; other: 90 reports, 9 with T
    rows = [(i, "X", "T") for i in range(6)] + [(i, "X", "U") for i in range(6, 10)]
    rows += [(i, "Y", "T") for i in range(10, 19)] + [(i, "Y", "U") for i in range(19, 100)]
    d = disproportionality(pd.DataFrame(rows, columns=["report_id", "product", "term"]))
    r = d[(d["product"] == "X") & (d["term"] == "T")].iloc[0]
    assert np.isclose(r["prr"], (6 / 10) / (9 / 90))
    assert np.isclose(r["ror"], (6 * 81) / (4 * 9))
    assert bool(r["signal"])


def test_compare_and_simulation_identity():
    rows = [(i, "X", "T") for i in range(6)] + [(i, "X", "U") for i in range(6, 10)]
    rows += [(i, "Y", "T") for i in range(10, 19)] + [(i, "Y", "U") for i in range(19, 100)]
    off = pd.DataFrame(rows, columns=["report_id", "product", "term"])
    ref = disproportionality(off)
    same = disproportionality(simulate_coding(off, {}, 1.0, 0.0, [], None))
    c = compare(ref, same)
    assert c["lost"] == 0 and c["gained"] == 0 and c["kept"] == c["ref_signals"]


def test_perturbations_change_text_deterministically():
    t = "The patient had shortness of breath and fever in the hospital with headache. " * 5
    for name, fn in PERTURBATIONS.items():
        assert fn(t, seed=1) == fn(t, seed=1)
        assert fn(t, seed=1) != t, name
