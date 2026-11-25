"""Test fixtures.

THE FIXTURE DATA HERE IS SYNTHETIC: tiny made-up reports in VAERS file format, used only to test code paths
offline. It is never used for any reported result. Real results come from the VAERS public CSVs.
"""
from __future__ import annotations

import json
import random

import pandas as pd
import pytest
import yaml

EVENTS = [("fever", "Pyrexia"), ("headache", "Headache"), ("injection site pain", "Injection site pain"),
          ("rash", "Rash"), ("nausea", "Nausea"), ("fatigue", "Fatigue"), ("chills", "Chills"),
          ("dizziness", "Dizziness"), ("myalgia", "Myalgia"), ("syncope", "Syncope")]
SERIOUS = [("was admitted to the hospital", "HOSPITAL"), ("died two days later", "DIED"),
           ("had anaphylaxis and needed epinephrine", "L_THREAT")]
VAX = [("COVID19", "COVID19 (COVID19 (MODERNA))", "MODERNA", "Moderna covid vaccine"),
       ("FLU4", "INFLUENZA (SEASONAL) (FLUZONE QUADRIVALENT)", "SANOFI PASTEUR", "flu shot"),
       ("VARZOS", "ZOSTER (SHINGRIX)", "GLAXOSMITHKLINE BIOLOGICALS", "Shingrix")]


def make_raw(dirpath, year=2023, n=160, seed=0):
    rng = random.Random(seed)
    data, sym, vax = [], [], []
    for i in range(n):
        vid = 1000 + i
        vt, vname, manu, mention = VAX[i % 3]
        evs = rng.sample(EVENTS, k=rng.randint(1, 3))
        if vt == "VARZOS" and ("rash", "Rash") not in evs:
            evs.append(("rash", "Rash"))        # gives the signal layer something disproportionate
        ser = SERIOUS[i % 3] if i % 4 == 0 else None
        text = f"Patient is a {20 + i % 60} year old who received the {mention}. Next day developed " + \
               " and ".join(e[0] for e in evs) + "."
        if ser:
            text += f" Patient {ser[0]}."
        row = {"VAERS_ID": vid, "RECVDATE": "01/15/2023", "STATE": "TX", "AGE_YRS": 20 + i % 60, "SEX": "F" if i % 2 else "M",
               "SYMPTOM_TEXT": text, "DIED": "", "L_THREAT": "", "ER_VISIT": "", "HOSPITAL": "", "HOSPDAYS": "",
               "X_STAY": "", "DISABLE": "", "RECOVD": "Y", "VAX_DATE": "01/10/2023", "ONSET_DATE": "01/11/2023",
               "NUMDAYS": "1", "BIRTH_DEFECT": "", "OFC_VISIT": "", "ER_ED_VISIT": ""}
        if ser:
            row[ser[1]] = "Y"
        data.append(row)
        s = {"VAERS_ID": vid}
        for j, (_, term) in enumerate(evs[:5], start=1):
            s[f"SYMPTOM{j}"], s[f"SYMPTOMVERSION{j}"] = term, "26.0"
        sym.append(s)
        vax.append({"VAERS_ID": vid, "VAX_TYPE": vt, "VAX_MANU": manu, "VAX_LOT": "AB12", "VAX_DOSE_SERIES": "1",
                    "VAX_ROUTE": "IM", "VAX_SITE": "LA", "VAX_NAME": vname})
    pd.DataFrame(data).to_csv(dirpath / f"{year}VAERSDATA.csv", index=False, encoding="latin-1")
    pd.DataFrame(sym).to_csv(dirpath / f"{year}VAERSSYMPTOMS.csv", index=False, encoding="latin-1")
    pd.DataFrame(vax).to_csv(dirpath / f"{year}VAERSVAX.csv", index=False, encoding="latin-1")


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throwaway project config pointing at synthetic raw files, loaded and sampled."""
    raw = tmp_path / "raw"
    raw.mkdir()
    make_raw(raw)
    cfg = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))
    cfg.update({"years": [2023], "db_path": str(tmp_path / "pv.duckdb"), "raw_dir": str(raw),
                "results_dir": str(tmp_path / "results")})
    cfg["sample"].update({"eval_size": 120, "llm_subset": 20})
    cfg["coding"]["min_term_freq"] = 1
    cp = tmp_path / "config.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("PV_CONFIG", str(cp))
    from pvagent.config import load_config
    load_config.cache_clear()
    from pvagent.data.load import load_vaers
    from pvagent.data.sample import build_sample
    from pathlib import Path
    load_vaers(raw, [2023], Path(cfg["db_path"]))
    build_sample(Path(cfg["db_path"]), 120, 0.5, 0.4, 42, tmp_path / "split.csv")
    yield cfg
    load_config.cache_clear()


class FakeLLM:
    """Deterministic stand-in for an LLM: reads the narrative and candidates and answers like a good model."""
    provider, model = "fake", "fake-1"

    def __init__(self):
        self.calls = 0

    def used_today(self):
        return 0

    def chat(self, messages, temperature=0.0, sample=0, json_mode=True):
        from pvagent.agent.llm import LLMResponse
        self.calls += 1
        u = messages[-1]["content"]
        narr = u.split('"""')[1]
        cands = []
        if "CANDIDATE TERMS:" in u:
            block = u.split("CANDIDATE TERMS:\n")[1].split("\n\n")[0]
            cands = [ln.split(": ", 1)[1] for ln in block.splitlines()]
        events = []
        for verb, term in EVENTS:
            if verb in narr.lower():
                ids = [cands.index(term)] if term in cands else []
                events.append({"verbatim": verb, "quote": verb, "term_ids": ids, "terms": [term], "onset_days": 1})
        serious = any(s[0] in narr for s in SERIOUS)
        out = {"patient": {"age_years": 30, "sex": "U", "quote": "year old"},
               "reporter": {"type": "unknown", "quote": ""},
               "products": [{"name": "vaccine", "vax_type": "COVID19" if "covid" in narr.lower() else None,
                             "quote": "received the"}],
               "events": events + [{"verbatim": "made up", "quote": "this text is not in the narrative", "term_ids": []}],
               "seriousness": {"serious": serious, "criteria": ["hospitalisation"] if "hospital" in narr else [],
                               "quotes": ["admitted to the hospital"] if "hospital" in narr else []},
               "outcome": "unknown", "confidence": 0.9}
        return LLMResponse(json.dumps(out), 100, 50, 0.01, False, self.model)
