"""The intake agent: one LLM call per case (extraction + seriousness + coding), then deterministic checks.

Pipeline per case:
  1. rule triage (deterministic, recall-first)            -> always runs, never skipped
  2. candidate retrieval from narrative clauses           -> coding is constrained to these terms
  3. LLM extraction with quote-level evidence + coding    -> JSON
  4. span verifier                                        -> drops fields whose quote is not in the text
  5. validity check (four minimum elements)
  6. expectedness vs DailyMed label section
  7. confidence + routing (auto vs human review)
Modes: "retrieval" (choose from candidates) or "prompt_only" (free-text terms, for the ablation).
Hybrid: run the small model; escalate to the large model when confidence < threshold.
"""
from __future__ import annotations

import json

from ..baselines.retrieval import TermIndex
from ..baselines.triage_rules import triage as rule_triage
from ..text import clauses
from .expectedness import ExpectednessChecker
from .llm import BudgetExceeded, LLMClient, LLMError, parse_json
from .schema import Event, Evidence, Patient, Product, Provenance, Reporter, SafetyCase, Seriousness, Validity
from .verifier import verify_case

PROMPT_VERSION = "intake-v2"
MAX_NARRATIVE_CHARS = 3500
VALID_CRITERIA = {"death", "life_threatening", "hospitalisation", "disability", "birth_defect",
                  "other_medically_important"}

SYSTEM = """You are a pharmacovigilance case-intake assistant. You convert one free-text adverse-event report
about a vaccine into a structured case. Rules:
- Copy evidence quotes EXACTLY from the narrative (verbatim substring, 2-15 words). Never paraphrase a quote.
- Only record what the narrative states. Do not infer causality. If something is not stated, use null/"unknown".
- Seriousness follows ICH E2A: death, life_threatening, hospitalisation (initial or prolonged), disability,
  birth_defect, other_medically_important. A visit to the ER alone is NOT hospitalisation.
- Be exhaustive when listing events: one entry for every distinct symptom, diagnosis, abnormal test result or
  product-use problem (e.g. storage error, wrong age, wrong dose, drug exposure during pregnancy). Typical reports
  have 3-6 coded terms. If one event fits several candidate terms, return all applicable term_ids.
- Coding conventions of the reference data (derived from the VAERS dataset): do NOT code hospitalisation, ER visit or
  other seriousness status as an event (that is captured under seriousness). DO code "Death" when the patient died,
  "SARS-CoV-2 test positive" (and "COVID-19" if COVID-19 illness is stated) when a positive test or infection is
  reported, and "No adverse event" when the narrative says no adverse event occurred.
- Reply with a single JSON object and nothing else."""

SCHEMA_RETRIEVAL = """{
 "patient": {"age_years": number|null, "sex": "F"|"M"|"U", "quote": str},
 "reporter": {"type": "patient"|"healthcare_professional"|"caregiver"|"other"|"unknown", "quote": str},
 "products": [{"name": str, "vax_type": str|null, "dose_number": str|null, "lot": str|null, "quote": str}],
 "events": [{"verbatim": str, "quote": str, "term_ids": [int], "onset_days": number|null}],
 "seriousness": {"serious": bool, "criteria": [str], "quotes": [str]},
 "outcome": "recovered"|"recovering"|"not_recovered"|"recovered_with_sequelae"|"fatal"|"unknown",
 "confidence": number between 0 and 1 (how sure you are the whole case is correct)
}
term_ids: ids from CANDIDATE TERMS that code this event (use [] if none fit). Do not invent ids.
vax_type: a VAERS code if obvious (e.g. COVID19, FLU4, VARZOS, HPV9, PNC20, TDAP, RSV, HEPB, MMR), else null."""

SCHEMA_PROMPT_ONLY = SCHEMA_RETRIEVAL.replace('"term_ids": [int]', '"terms": [str]').replace(
    "term_ids: ids from CANDIDATE TERMS that code this event (use [] if none fit). Do not invent ids.",
    "terms: standard medical preferred terms (MedDRA-style wording, e.g. 'Pyrexia', 'Injection site pain').")


def _candidates(index: TermIndex, text: str, k: int, n_prior: int = 0) -> list[str]:
    qs = clauses(text)[:50] or [text]
    return index.candidates(qs, k=k, n_prior=n_prior)


def build_messages(narrative: str, candidates: list[str] | None, structured_hint: dict | None = None) -> list[dict]:
    text = narrative[:MAX_NARRATIVE_CHARS]
    parts = [f"NARRATIVE:\n\"\"\"\n{text}\n\"\"\""]
    if structured_hint:
        parts.append("STRUCTURED FORM FIELDS (from the report form, may be blank):\n" + json.dumps(structured_hint))
    if candidates is not None:
        parts.append("CANDIDATE TERMS:\n" + "\n".join(f"{i}: {t}" for i, t in enumerate(candidates)))
        parts.append("Return JSON with this shape:\n" + SCHEMA_RETRIEVAL)
    else:
        parts.append("Return JSON with this shape:\n" + SCHEMA_PROMPT_ONLY)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def _num(x):
    try:
        return float(x) if x is not None and x != "" else None
    except (TypeError, ValueError):
        return None


def parse_case(case_id: int, narrative: str, d: dict, candidates: list[str] | None) -> SafetyCase:
    p = d.get("patient") or {}
    r = d.get("reporter") or {}
    sex = str(p.get("sex") or "U").upper()[:1]
    rtype = r.get("type") if r.get("type") in {"patient", "healthcare_professional", "caregiver", "other"} else "unknown"
    case = SafetyCase(
        case_id=case_id, narrative=narrative,
        patient=Patient(age_years=_num(p.get("age_years")), sex=sex if sex in "FMU" else "U",
                        evidence=Evidence(quote=p.get("quote") or "")),
        reporter=Reporter(type=rtype, evidence=Evidence(quote=r.get("quote") or "")),
    )
    for pr in d.get("products") or []:
        if not isinstance(pr, dict) or not pr.get("name"):
            continue
        case.products.append(Product(name=str(pr["name"]), vax_type=pr.get("vax_type"),
                                     dose_number=None if pr.get("dose_number") is None else str(pr.get("dose_number")),
                                     lot=pr.get("lot"), evidence=Evidence(quote=pr.get("quote") or "")))
    invented = 0
    for ev in d.get("events") or []:
        if not isinstance(ev, dict) or not (ev.get("verbatim") or ev.get("quote")):
            continue
        verb = str(ev.get("verbatim") or ev.get("quote"))
        terms: list[str] = []
        if candidates is not None:
            for tid in ev.get("term_ids") or []:
                try:
                    terms.append(candidates[int(tid)])
                except (ValueError, IndexError, TypeError):
                    invented += 1                       # id outside the list -> rejected
        else:
            terms = [str(t) for t in ev.get("terms") or [] if t]
        for t in terms or [None]:
            case.events.append(Event(verbatim=verb, coded_term=t, onset_days=_num(ev.get("onset_days")),
                                     evidence=Evidence(quote=ev.get("quote") or verb)))
    s = d.get("seriousness") or {}
    crit = [c for c in (s.get("criteria") or []) if c in VALID_CRITERIA]
    case.seriousness = Seriousness(serious=bool(s.get("serious")) or bool(crit), criteria=crit,
                                   evidence=[Evidence(quote=q) for q in (s.get("quotes") or []) if isinstance(q, str)])
    out = d.get("outcome")
    case.outcome = out if out in {"recovered", "recovering", "not_recovered", "recovered_with_sequelae", "fatal"} else "unknown"
    case.confidence_parts["self"] = max(0.0, min(1.0, _num(d.get("confidence")) or 0.0))
    if invented:
        case.rejected_fields.append(f"invalid_term_ids:{invented}")
    return case


def _jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / max(1, len(a | b))


class IntakeAgent:
    def __init__(self, small: LLMClient, index: TermIndex, expect: ExpectednessChecker | None = None,
                 large: LLMClient | None = None, mode: str = "retrieval", n_candidates: int = 25,
                 threshold: float = 0.8, consistency_k: int = 0, temperature: float = 0.0, n_prior: int = 0,
                 lr_scores: dict | None = None, lr_threshold: float = 1.1):
        self.small, self.large, self.index, self.expect = small, large, index, expect
        self.mode, self.k, self.threshold = mode, n_candidates, threshold
        self.consistency_k, self.temperature, self.n_prior = consistency_k, temperature, n_prior
        self.lr_scores, self.lr_threshold = lr_scores or {}, lr_threshold

    def _run_once(self, client: LLMClient, cid: int, narrative: str, hint: dict | None,
                  temperature: float, sample: int):
        cands = _candidates(self.index, narrative, self.k, self.n_prior) if self.mode == "retrieval" else None
        msgs = build_messages(narrative, cands, hint)
        resp = client.chat(msgs, temperature=temperature, sample=sample)
        case = parse_case(cid, narrative, parse_json(resp.text), cands)
        return case, resp

    def process(self, cid: int, narrative: str, hint: dict | None = None, vax_types_for_label: list[str] | None = None,
                client: LLMClient | None = None, form: dict | None = None) -> SafetyCase:
        client = client or self.small
        rules = rule_triage(narrative)
        lr_p = self.lr_scores.get(cid)
        lr_pos = lr_p is not None and lr_p >= self.lr_threshold
        try:
            case, resp = self._run_once(client, cid, narrative, hint, self.temperature, 0)
        except BudgetExceeded:
            raise
        except (LLMError, ValueError, json.JSONDecodeError) as e:
            case = SafetyCase(case_id=cid, narrative=narrative, error=f"{type(e).__name__}: {e}"[:300])
            case.seriousness.rule_serious, case.seriousness.rule_criteria = rules.serious, sorted(rules.criteria)
            case.seriousness.lr_serious, case.seriousness.lr_score = (lr_pos if lr_p is not None else None), lr_p
            case.route, case.route_reasons = "human_review", ["llm_error"]
            return case
        case.provenance = Provenance(model=resp.model, prompt_version=PROMPT_VERSION, mode=self.mode,
                                     prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                                     seconds=resp.seconds, cached=resp.cached)
        case.rejected_fields += verify_case(case, narrative)
        case.seriousness.rule_serious = rules.serious
        case.seriousness.rule_criteria = sorted(rules.criteria)
        case.seriousness.lr_serious, case.seriousness.lr_score = (lr_pos if lr_p is not None else None), lr_p

        # 5. validity: VAERS public data redacts reporter identity, but every report comes from an identified
        # submitter on the VAERS form, so reporter identifiability is satisfied by the channel.
        # `form` = structured VAERS form fields used ONLY for the validity check (never shown to the model, and
        # never the evaluated truths: seriousness flags, coded terms, vaccine table).
        form = form or {}
        case.validity = Validity(
            identifiable_patient=case.patient.age_years is not None or case.patient.sex != "U"
            or bool(form.get("age") is not None and form.get("age") == form.get("age")) or form.get("sex") in ("F", "M"),
            identifiable_reporter=True,
            suspect_product=bool(case.products) or bool(form.get("has_vaccine")),
            adverse_event=bool(case.events),
        )
        # 6. expectedness
        if self.expect is not None and self.expect.available:
            vts = vax_types_for_label or [p.vax_type for p in case.products if p.vax_type]
            for e in case.events:
                if e.coded_term:
                    e.expectedness, e.label_match = self.expect.check(e.coded_term, vts)

        # 7. confidence
        n_ev = 1 + len(case.events) + len(case.products) + len(case.seriousness.evidence)
        n_bad = len([f for f in case.rejected_fields if not f.startswith("invalid_term_ids")])
        case.confidence_parts["span"] = max(0.0, 1 - n_bad / n_ev)
        rule_pos = rules.score == 1.0 or lr_pos      # deterministic layers: keyword rules OR logistic regression
        case.confidence_parts["triage_agree"] = 1.0 if rule_pos == case.seriousness.serious else 0.0
        if self.consistency_k > 0:
            terms0 = {e.coded_term for e in case.events if e.coded_term}
            agree_t, agree_s = [], []
            for i in range(1, self.consistency_k + 1):
                try:
                    alt, _ = self._run_once(client, cid, narrative, hint, 0.7, i)
                except (LLMError, ValueError, json.JSONDecodeError):
                    continue
                agree_t.append(_jaccard(terms0, {e.coded_term for e in alt.events if e.coded_term}))
                agree_s.append(1.0 if alt.seriousness.serious == case.seriousness.serious else 0.0)
            if agree_t:
                case.confidence_parts["consistency"] = 0.5 * sum(agree_t) / len(agree_t) + 0.5 * sum(agree_s) / len(agree_s)
        w = {"self": 0.3, "span": 0.3, "triage_agree": 0.2, "consistency": 0.4}
        num = sum(w[k] * v for k, v in case.confidence_parts.items() if k in w)
        den = sum(w[k] for k in case.confidence_parts if k in w)
        case.confidence = round(num / den, 4) if den else 0.0

        # routing
        reasons = []
        if case.confidence < self.threshold:
            reasons.append("low_confidence")
        if rule_pos != case.seriousness.serious:
            reasons.append("triage_disagreement")
        if not case.validity.valid:
            reasons.append("invalid_icsr")
        if not case.events or any(e.coded_term is None for e in case.events):
            reasons.append("uncoded_event")
        case.route_reasons = reasons
        case.route = "human_review" if reasons else "auto"

        # hybrid escalation
        if self.large is not None and client is self.small and case.confidence < self.threshold:
            big = self.process(cid, narrative, hint, vax_types_for_label, client=self.large, form=form)
            big.provenance.escalated = True
            big.provenance.prompt_tokens += case.provenance.prompt_tokens
            big.provenance.completion_tokens += case.provenance.completion_tokens
            return big
        return case
