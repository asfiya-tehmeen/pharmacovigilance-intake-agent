"""Human review queue for agent-processed safety cases.

Local:   pv review   (or: streamlit run app/review_app.py)
         Shows the demo set plus your own results/<run>/ folders; decisions are appended to <run>/reviews.jsonl.
Public:  on Streamlit Community Cloud (or with PV_DEMO_ONLY=1) only the bundled demo set in app/demo/ is shown, and
         decisions stay in the visitor's own session (downloadable as JSON). Nothing is written to disk.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pvagent.agent.schema import SafetyCase  # noqa: E402

DEMO_DIR = ROOT / "app" / "demo"
DEMO_ONLY = os.environ.get("PV_DEMO_ONLY") == "1" or str(ROOT).replace("\\", "/").startswith("/mount/src")
DEMO_LABEL = "Demo set (VAERS test cases)"
REPO_URL = "https://github.com/asfiya-tehmeen/pharmacovigilance-intake-agent"
CRITERIA = ["death", "life_threatening", "hospitalisation", "disability", "birth_defect", "other_medically_important"]
st.set_page_config(page_title="PV case-intake review", layout="wide")


def highlight(text: str, quotes: list[tuple[str, str]]) -> str:
    out = html.escape(text)
    for q, color in sorted(quotes, key=lambda x: -len(x[0])):
        if len(q) < 3:
            continue
        pat = re.compile(re.escape(html.escape(q)), re.I)
        out = pat.sub(lambda m: f'<mark style="background:{color};padding:0 2px;border-radius:3px">{m.group(0)}</mark>',
                      out, count=1)
    return f'<div style="white-space:pre-wrap;line-height:1.6;font-size:0.95rem">{out}</div>'


def run_dirs() -> dict[str, Path]:
    out = {}
    if (DEMO_DIR / "cases.jsonl").exists():
        out[DEMO_LABEL] = DEMO_DIR
    if not DEMO_ONLY:
        for p in sorted((ROOT / "results").glob("*"), reverse=True):
            f = p / "cases.jsonl"
            if f.exists() and f.stat().st_size > 0:
                out[p.name] = p
    return out


@st.cache_data
def load_cases(run_dir: str) -> list[dict]:
    return [json.loads(l) for l in open(Path(run_dir) / "cases.jsonl", encoding="utf-8") if l.strip()]


@st.cache_data
def load_official(run_dir: str) -> dict[int, dict]:
    """Official VAERS coding for comparison: bundled for the demo; read from the local database otherwise."""
    p = Path(run_dir) / "official.json"
    if p.exists():
        return {int(k): v for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    try:
        import duckdb
        from pvagent.config import load_config, path
        from pvagent.data.sample import get_cases
        ids = [c["case_id"] for c in load_cases(run_dir)]
        con = duckdb.connect(str(path(load_config()["db_path"])), read_only=True)
        g = get_cases(con, ids=ids)
        con.close()
        return {int(r.VAERS_ID): {"serious": bool(r.SERIOUS), "terms": sorted(r.terms), "vax_types": sorted(set(r.vax_types))}
                for r in g.itertuples()}
    except Exception:
        return {}


@st.cache_resource
def db_vocabulary() -> list[str]:
    if DEMO_ONLY:
        return []
    try:
        import duckdb
        from pvagent.config import load_config, path
        from pvagent.data.sample import term_vocabulary
        con = duckdb.connect(str(path(load_config()["db_path"])), read_only=True)
        v = term_vocabulary(con, load_config()["coding"]["min_term_freq"])["TERM"].tolist()
        con.close()
        return v
    except Exception:
        return []


def reviews_for(run_dir: Path) -> dict[int, dict]:
    if DEMO_ONLY or run_dir == DEMO_DIR:
        return st.session_state.setdefault(f"reviews::{run_dir}", {})
    p = run_dir / "reviews.jsonl"
    if not p.exists():
        return {}
    return {r["case_id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}


def save_review(run_dir: Path, rec: dict):
    if DEMO_ONLY or run_dir == DEMO_DIR:
        st.session_state.setdefault(f"reviews::{run_dir}", {})[rec["case_id"]] = rec
    else:
        with open(run_dir / "reviews.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def flagged(c: dict) -> bool:
    s = c["seriousness"]
    return bool(s["serious"] or s["rule_serious"] or s.get("lr_serious"))


# ------------------------------------------------------------------ sidebar
runs = run_dirs()
st.sidebar.title("PV case-intake review")
st.sidebar.caption("Decision support only. No causality assessment. Not for clinical use.")
if not runs:
    st.info("No agent runs found. Run `pv agent --limit 20` first.")
    st.stop()
run_name = st.sidebar.selectbox("Run", list(runs))
run_dir = runs[run_name]
cases = load_cases(str(run_dir))
official = load_official(str(run_dir))
reviews = reviews_for(run_dir)
only_queue = st.sidebar.toggle("Only human-review queue", value=False)
only_open = st.sidebar.toggle("Hide reviewed", value=False)
sort = st.sidebar.radio("Order", ["Serious first, then lowest confidence", "Lowest confidence", "Case ID"])

q = [c for c in cases if (not only_queue or c["route"] == "human_review") and (not only_open or c["case_id"] not in reviews)]
if sort.startswith("Serious"):
    q.sort(key=lambda c: (not flagged(c), c["confidence"]))
elif sort == "Lowest confidence":
    q.sort(key=lambda c: c["confidence"])
else:
    q.sort(key=lambda c: c["case_id"])

c1, c2, c3, c4 = st.sidebar.columns(2) + st.sidebar.columns(2)
c1.metric("Cases", len(cases))
c2.metric("Review queue", sum(c["route"] == "human_review" for c in cases))
c3.metric("Reviewed", len(reviews))
c4.metric("Shown", len(q))
if reviews:
    st.sidebar.download_button("Download my decisions (JSON)", json.dumps(list(reviews.values()), indent=2),
                               file_name=f"reviews_{run_name.replace(' ', '_').replace(':', '')}.json",
                               mime="application/json")

# ------------------------------------------------------------------ header
st.title("Pharmacovigilance case-intake agent")
with st.expander("What is this?", expanded=run_dir == DEMO_DIR):
    st.markdown(
        "An LLM agent turns free-text vaccine adverse-event reports (real public **VAERS** narratives) into structured, "
        "coded safety cases with a verbatim evidence quote for every field, and routes uncertain cases to a human. "
        "This page is the human review queue. Highlighted text is the evidence the agent cited.\n\n"
        "**Measured on 300 held-out reports:** seriousness recall 98.7% (agent + keyword rules + logistic regression) "
        "at 44% review workload; event-coding agreement with official coding F1 0.52 vs 0.34 for a dictionary baseline. "
        f"Full method, limits and results: [REPORT.md]({REPO_URL}/blob/main/REPORT.md).\n\n"
        "VAERS reports are unverified and cannot establish that a vaccine caused an event. Nothing here is a safety "
        "conclusion. Decisions you make on this page stay in your browser session.")
if not q:
    st.success("Nothing to show with these filters.")
    st.stop()

labels = [f"{c['case_id']}  ·  {'SERIOUS' if flagged(c) else 'non-serious'}  ·  conf {c['confidence']:.2f}"
          + ("  ·  reviewed" if c["case_id"] in reviews else "") for c in q]
idx = st.sidebar.selectbox("Case", range(len(q)), format_func=lambda i: labels[i])
c = SafetyCase.model_validate(q[idx])

# ------------------------------------------------------------------ case
st.subheader(f"VAERS report {c.case_id}")
top = st.columns(5)
top[0].metric("Confidence", f"{c.confidence:.2f}")
top[1].metric("LLM: serious", "yes" if c.seriousness.serious else "no")
top[2].metric("Rules: serious", "yes" if c.seriousness.rule_serious else "no")
top[3].metric("Regression: serious", "—" if c.seriousness.lr_serious is None else ("yes" if c.seriousness.lr_serious else "no"))
top[4].metric("Valid report", "yes" if c.validity.valid else "no")
if c.route == "human_review":
    st.warning("Routed to human review: " + ", ".join(c.route_reasons))
else:
    st.info("Auto-processed (no routing rule fired).")
if c.error:
    st.error(c.error)

left, right = st.columns([3, 2])
with left:
    quotes = [(e.evidence.quote, "#ffe08a") for e in c.events] + [(p.evidence.quote, "#b9e3ff") for p in c.products]
    quotes += [(e.quote, "#ffb3b3") for e in c.seriousness.evidence]
    st.markdown("**Narrative** <span style='background:#ffe08a'>event</span> "
                "<span style='background:#b9e3ff'>product</span> <span style='background:#ffb3b3'>seriousness</span>",
                unsafe_allow_html=True)
    st.markdown(highlight(c.narrative, quotes), unsafe_allow_html=True)
with right:
    st.markdown("**Extracted by the agent**")
    st.write(f"Patient: age {c.patient.age_years}, sex {c.patient.sex} · Reporter: {c.reporter.type} · Outcome: {c.outcome}")
    st.write("Products: " + (", ".join(f"{p.name} ({p.vax_type or 'code not given'})" for p in c.products) or "none found in text"))
    st.dataframe([{"reporter's words": e.verbatim, "coded term": e.coded_term, "onset (days)": e.onset_days,
                   "on label?": e.expectedness} for e in c.events], width="stretch", hide_index=True)
    if c.seriousness.criteria or c.seriousness.rule_criteria:
        st.caption("Seriousness criteria: LLM " + (", ".join(c.seriousness.criteria) or "none")
                   + " · rules " + (", ".join(c.seriousness.rule_criteria) or "none"))
    if c.rejected_fields:
        st.caption("Dropped by the evidence check (quote not in text): " + ", ".join(c.rejected_fields))
    st.caption("Confidence parts: " + ", ".join(f"{k}={v:.2f}" for k, v in c.confidence_parts.items()))
    off = official.get(c.case_id)
    if off:
        with st.expander("Official VAERS coding (used as the evaluation reference)"):
            agent_terms = {e.coded_term for e in c.events if e.coded_term}
            st.write("Serious (form flags): " + ("yes" if off["serious"] else "no"))
            st.write("Vaccine type: " + ", ".join(off.get("vax_types", [])))
            st.dataframe([{"official term": t, "agent also coded it": t in agent_terms} for t in off["terms"]]
                         + [{"official term": f"(agent only) {t}", "agent also coded it": False}
                            for t in sorted(agent_terms - set(off["terms"]))],
                         width="stretch", hide_index=True)

# ------------------------------------------------------------------ decision
st.divider()
st.markdown("**Reviewer decision**")
prev = reviews.get(c.case_id)
if prev:
    st.caption(f"Already reviewed: {prev['decision']} ({prev['ts'][:16]}). Saving again replaces it.")
with st.form(f"rev_{run_name}_{c.case_id}"):
    serious = st.checkbox("Serious", value=bool(prev["serious"]) if prev else flagged(q[idx]))
    crit = st.multiselect("Seriousness criteria", CRITERIA,
                          default=prev["criteria"] if prev else
                          [x for x in set(c.seriousness.criteria) | set(c.seriousness.rule_criteria) if x in CRITERIA])
    current = sorted({e.coded_term for e in c.events if e.coded_term})
    vocab = db_vocabulary() or sorted({e["coded_term"] for k in cases for e in k["events"] if e["coded_term"]}
                                      | {t for o in official.values() for t in o["terms"]})
    terms = st.multiselect("Coded event terms", sorted(set(vocab) | set(current)),
                           default=prev["terms"] if prev else current)
    decision = st.radio("Decision", ["accept", "accept_with_corrections", "reject_invalid"], horizontal=True)
    note = st.text_input("Note")
    reviewer = st.text_input("Reviewer initials", max_chars=8)
    if st.form_submit_button("Save decision", type="primary"):
        save_review(run_dir, {"case_id": c.case_id, "decision": decision, "serious": serious, "criteria": crit,
                              "terms": terms, "agent_terms": current, "agent_serious": c.seriousness.serious,
                              "note": note, "reviewer": reviewer, "ts": datetime.now().isoformat()})
        st.success("Saved.")
        st.rerun()
