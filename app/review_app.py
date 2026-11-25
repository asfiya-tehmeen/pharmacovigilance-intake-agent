"""Human review queue for agent-processed safety cases.

Run:  pv review     (or: streamlit run app/review_app.py)
Reviewer decisions are appended to results/<run>/reviews.jsonl.
"""
from __future__ import annotations

import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pvagent.agent.schema import SafetyCase  # noqa: E402

CRITERIA = ["death", "life_threatening", "hospitalisation", "disability", "birth_defect", "other_medically_important"]
st.set_page_config(page_title="PV intake review", layout="wide")


def highlight(text: str, quotes: list[tuple[str, str]]) -> str:
    out = html.escape(text)
    for q, color in sorted(quotes, key=lambda x: -len(x[0])):
        if len(q) < 3:
            continue
        pat = re.compile(re.escape(html.escape(q)), re.I)
        out = pat.sub(lambda m: f'<mark style="background:{color};padding:0 2px;border-radius:3px">{m.group(0)}</mark>', out, count=1)
    return f'<div style="white-space:pre-wrap;line-height:1.6;font-size:0.95rem">{out}</div>'


@st.cache_data
def load_cases(run: str) -> list[dict]:
    return [json.loads(l) for l in open(ROOT / "results" / run / "cases.jsonl", encoding="utf-8") if l.strip()]


def load_reviews(run: str) -> dict[int, dict]:
    p = ROOT / "results" / run / "reviews.jsonl"
    if not p.exists():
        return {}
    revs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return {r["case_id"]: r for r in revs}


@st.cache_resource
def vocabulary() -> list[str]:
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


runs = sorted([p.name for p in (ROOT / "results").glob("*") if (p / "cases.jsonl").exists() and (p / "cases.jsonl").stat().st_size > 0], reverse=True)
st.sidebar.title("PV intake review")
st.sidebar.caption("Decision support only. No causality assessment. Not for clinical use.")
if not runs:
    st.info("No agent runs yet. Run `pv agent --limit 20` first.")
    st.stop()
run = st.sidebar.selectbox("Run", runs)
cases = load_cases(run)
reviews = load_reviews(run)
only_queue = st.sidebar.toggle("Only human-review queue", value=True)
only_open = st.sidebar.toggle("Hide reviewed", value=True)
sort = st.sidebar.radio("Order", ["Serious first, then lowest confidence", "Lowest confidence", "Case ID"])

q = [c for c in cases if (not only_queue or c["route"] == "human_review") and (not only_open or c["case_id"] not in reviews)]
if sort.startswith("Serious"):
    q.sort(key=lambda c: (not (c["seriousness"]["serious"] or c["seriousness"]["rule_serious"]), c["confidence"]))
elif sort == "Lowest confidence":
    q.sort(key=lambda c: c["confidence"])
else:
    q.sort(key=lambda c: c["case_id"])

n_h = sum(c["route"] == "human_review" for c in cases)
c1, c2, c3, c4 = st.sidebar.columns(2) + st.sidebar.columns(2)
c1.metric("Cases", len(cases))
c2.metric("Queue", n_h)
c3.metric("Reviewed", len(reviews))
c4.metric("Open", len(q))
if not q:
    st.success("Queue empty.")
    st.stop()

labels = [f"{c['case_id']}  ·  {'SERIOUS' if (c['seriousness']['serious'] or c['seriousness']['rule_serious']) else 'non-serious'}"
          f"  ·  conf {c['confidence']:.2f}" for c in q]
idx = st.sidebar.selectbox("Case", range(len(q)), format_func=lambda i: labels[i])
c = SafetyCase.model_validate(q[idx])

st.subheader(f"VAERS {c.case_id}")
top = st.columns(4)
top[0].metric("Confidence", f"{c.confidence:.2f}")
top[1].metric("Agent: serious", "yes" if c.seriousness.serious else "no")
top[2].metric("Rules: serious", "yes" if c.seriousness.rule_serious else "no")
top[3].metric("Valid ICSR", "yes" if c.validity.valid else "no")
if c.route_reasons:
    st.warning("Routed to review: " + ", ".join(c.route_reasons))
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
    st.markdown("**Extracted**")
    st.write(f"Patient: age {c.patient.age_years}, sex {c.patient.sex} · Reporter: {c.reporter.type} · Outcome: {c.outcome}")
    st.write("Products: " + (", ".join(f"{p.name} ({p.vax_type})" for p in c.products) or "—"))
    st.dataframe([{"verbatim": e.verbatim, "coded": e.coded_term, "onset_d": e.onset_days, "label": e.expectedness}
                  for e in c.events], use_container_width=True, hide_index=True)
    if c.rejected_fields:
        st.caption("Dropped by span verifier: " + ", ".join(c.rejected_fields))
    st.caption("Confidence parts: " + ", ".join(f"{k}={v:.2f}" for k, v in c.confidence_parts.items()))

st.divider()
st.markdown("**Reviewer decision**")
with st.form(f"rev_{c.case_id}"):
    serious = st.checkbox("Serious", value=bool(c.seriousness.serious or c.seriousness.rule_serious))
    crit = st.multiselect("Seriousness criteria", CRITERIA,
                          default=[x for x in set(c.seriousness.criteria) | set(c.seriousness.rule_criteria) if x in CRITERIA])
    vocab = vocabulary()
    current = sorted({e.coded_term for e in c.events if e.coded_term})
    terms = st.multiselect("Coded event terms", sorted(set(vocab) | set(current)) if vocab else current, default=current)
    decision = st.radio("Decision", ["accept", "accept_with_corrections", "reject_invalid"], horizontal=True)
    note = st.text_input("Note")
    reviewer = st.text_input("Reviewer initials", max_chars=8)
    if st.form_submit_button("Save decision", type="primary"):
        rec = {"case_id": c.case_id, "decision": decision, "serious": serious, "criteria": crit, "terms": terms,
               "agent_terms": current, "agent_serious": c.seriousness.serious, "note": note, "reviewer": reviewer,
               "ts": datetime.now().isoformat()}
        with open(ROOT / "results" / run / "reviews.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        st.success("Saved.")
        st.rerun()
