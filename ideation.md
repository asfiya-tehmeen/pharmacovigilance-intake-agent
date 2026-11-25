# F. Pharmacovigilance Case-Intake Agent — Ideation

**Status:** Ideation | **Data:** REAL (VAERS public reports; DailyMed labels; openFDA) | **Target roles:** AI engineer (pharma, CRO, drug safety tech), forward-deployed AI engineer, regulatory/PV automation

**One-line pitch:** An agent that turns free-text adverse-event reports into structured, coded safety cases, routes the uncertain ones to a human, and measures whether its mistakes would change which safety signals a company detects.

---

## 1. Why this matters

- Pharma, CROs and consultancies are publicly pushing agentic AI into PV case intake (Capgemini, IQVIA, TCS and Tech Mahindra all publish on it). IQVIA describes case processing as about 5 days with many handovers. These are vendor claims, so treat the numbers as marketing until verified.
- Regulators (FDA, EMA, SFDA's Saudi Vigilance, UAE's drug establishment) all require case processing with human accountability. TCS explicitly argues humans must stay accountable for safety-critical decisions.
- Most demos stop at "LLM extracts fields". Almost nobody measures the **downstream effect**: does extraction error change signal detection? That is the gap this project fills.

## 2. How the idea was improved from v1

| v1 idea | Improved version | Why |
|---|---|---|
| Extract fields from narratives | Full intake pipeline: validity check → triage → coding → expectedness → structured case → human queue | Mirrors how a real PV department works, so it reads as FDE-ready |
| Hand-label ~300 narratives | Use VAERS's own structured fields as ground truth | Real labels at scale with no hand-labelling for most tasks |
| Measure extraction F1 | Also measure **end-to-end signal impact** (do LLM-coded cases produce the same disproportionality signals as officially coded ones?) | Novel, decision-relevant question |
| One model | Compare LLM vs classical baseline vs hybrid, with calibrated abstention | Shows engineering judgement and cost awareness |
| English only | Robustness tests: typos, abbreviations, lay language, optional translation round-trip | Real intake is messy; Gulf intake is bilingual |

## 3. Research questions

1. **Extraction:** How accurately can an LLM extract suspect product, events, onset, seriousness and outcome from real narratives, compared with a dictionary/NER baseline?
2. **Coding:** How well can it map free-text events to the standard terms used in the structured fields?
3. **Triage:** Can it flag serious cases (death, life-threatening, hospitalisation, disability) with very high recall? What does that recall cost in human review workload?
4. **Calibration:** Does model confidence predict correctness well enough to route only uncertain cases to humans?
5. **Downstream impact (headline question):** If cases are coded by the agent instead of the official coding, how many disproportionality signals (PRR/ROR) are gained, lost or shifted?
6. **Robustness:** How much does accuracy drop under realistic noise (typos, abbreviations, lay phrasing, translation)?

## 4. Data plan (real first)

| Need | Source | Real? | Notes / to verify |
|---|---|---|---|
| Free-text narratives | VAERS public CSVs (`VAERSDATA` symptom text) | ✅ Real | US reports keep narratives; foreign reports have free text removed. Read the VAERS Data Use Guide and terms before use. |
| Ground truth: seriousness | VAERS structured flags (died, life-threatening, hospitalised, disability, ER visit, etc.) | ✅ Real | Field names to confirm in the data guide. |
| Ground truth: events | VAERS coded symptom terms (`VAERSSYMPTOMS`) | ✅ Real | I believe these are standard coded terms; confirm in the data guide. Do not redistribute any licensed dictionary; only use the terms as they appear in the data. |
| Ground truth: product | VAERS vaccine table (`VAERSVAX`) | ✅ Real | Vaccine type, manufacturer, lot, dose. |
| Expectedness (is the event listed in the label?) | DailyMed vaccine/drug labels (free API) | ✅ Real | Parse adverse-reactions sections; check label versions by date. |
| Drug (non-vaccine) context | openFDA drug event API (FAERS structured data) | ✅ Real | No narratives, so use it for the signal-detection comparison and coding vocabulary only. |
| Noise/robustness tests | Perturbations applied to real narratives | Real data, synthetic noise | Clearly labelled as perturbed. |

Synthetic data: **none needed** for the core. Only the robustness perturbations are synthetic.

Scope to keep it tractable: 2–3 report years, a sample of 5,000–20,000 reports for evaluation (stratified so serious cases are well represented), full data for signal comparison where only cheap baselines run.

## 5. System design

```
VAERS CSVs ──► loader + dedupe ──► case store (DuckDB)
                                      │
             ┌────────────────────────┼───────────────────────────┐
             ▼                        ▼                           ▼
   1. Validity check         2. Extraction agent          3. Seriousness triage
   (4 minimum ICSR           (product, events, onset,     (rules + LLM, recall-first)
   elements: patient,        outcome, dose, lot) with
   reporter, product,        quote-level evidence spans
   event)                            │
                                     ▼
                         4. Event coding (retrieve candidate terms → LLM picks → verifier)
                                     │
                                     ▼
                         5. Expectedness check vs DailyMed label section
                                     │
                                     ▼
                         6. Structured case output (E2B(R3)-inspired JSON schema)
                                     │
                     confidence ≥ threshold? ──no──► Human review queue (Streamlit)
                                     │yes
                                     ▼
                         7. Signal layer: PRR/ROR on agent-coded vs official coding
```

Key engineering choices:
- **Evidence spans:** every extracted field must quote the narrative text it came from; a deterministic checker rejects fields whose quote is not in the narrative.
- **Retrieval-constrained coding:** the model chooses from retrieved candidate terms rather than inventing terms.
- **Validity rule:** follows the widely used "four minimum elements" for a valid individual case safety report (identifiable patient, identifiable reporter, suspect product, adverse event).
- **Recall-first triage:** a deterministic keyword/rule layer runs before the LLM so serious cases are never dropped silently.

## 6. Evaluation plan

| Task | Ground truth | Metrics | Baselines |
|---|---|---|---|
| Seriousness triage | VAERS flags | Recall (primary), precision, review workload at 99% recall | Keyword rules; logistic regression on TF-IDF |
| Event extraction + coding | VAERS coded symptoms | Micro/macro F1 per case, exact-match rate | Dictionary match; TF-IDF nearest term |
| Product extraction | VAERS vaccine table | Accuracy | Regex/dictionary |
| Calibration | Correct/incorrect per field | Expected calibration error, reliability curves, risk–coverage curve | Raw model confidence vs self-consistency vs verifier score |
| Signal impact | PRR/ROR from official coding | Signals gained/lost, rank correlation of signal lists, change in top-50 | Official coding = reference |
| Robustness | Same labels, perturbed text | Accuracy drop per perturbation type | — |
| Cost | Token logs | Tokens and seconds per case; cost at paid rates | — |

Experiments to run:
1. Prompt-only vs retrieval-constrained coding.
2. Small model vs large model vs hybrid (small model first, escalate when uncertain).
3. Threshold sweep: human workload vs missed serious cases.
4. Signal impact at different extraction error levels.

## 7. Free stack and budget

- Python, pandas/polars, DuckDB, scikit-learn, rapidfuzz, Pydantic, Streamlit, pytest.
- LLMs: Groq free tier (one secondary source reports about 200K tokens/day and 1,000 requests/day for a large open model; check your console), Gemini free tier as backup, or a small local model via Ollama if your laptop can run it.
- Budget implication: about 300–600 narratives/day on the free tier at roughly 500 tokens per case. So: run cheap baselines on everything, run the LLM on a stratified sample, cache every response, and spread evaluation runs over several days.

## 8. Step-by-step plan

### Phase 0 — Decide (½ day)
- [ ] Read the VAERS Data Use Guide and terms; note restrictions in `notes/data-terms.md`.
- [ ] Choose years, sample size and the serious-case stratification.
- [ ] Decide the LLM provider(s) and check today's free-tier limits.

### Phase 1 — Data (1–2 days)
- [ ] Download VAERS CSVs for chosen years; load into DuckDB.
- [ ] Join narratives, symptoms and vaccine tables; document duplicates and missing narratives.
- [ ] Build the evaluation sample and a frozen test split.
- [ ] Pull relevant DailyMed label sections for the products in scope.

### Phase 2 — Baselines (1 day)
- [ ] Keyword/rule seriousness triage.
- [ ] Dictionary and TF-IDF event coding.
- [ ] Logistic regression triage baseline.

### Phase 3 — Agent (3–4 days)
- [ ] Pydantic schema for the structured case (E2B(R3)-inspired, simplified).
- [ ] Extraction with evidence spans + deterministic span verifier.
- [ ] Retrieval-constrained event coding.
- [ ] Expectedness check against label text.
- [ ] Confidence scoring (self-consistency and verifier agreement).
- [ ] Human review queue in Streamlit.

### Phase 4 — Evaluation (2–3 days)
- [ ] Run all metrics in section 6; save results with run IDs.
- [ ] Threshold sweep and risk–coverage curves.
- [ ] Signal-impact comparison.
- [ ] Robustness perturbations.

### Phase 5 — Write-up and ship (1–2 days)
- [ ] `REPORT.md`: findings first, including failures.
- [ ] Short Loom demo; deploy the review UI (free hosting).
- [ ] LinkedIn post on the signal-impact finding.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| VAERS coded terms may not match narrative wording exactly | Report agreement, not "accuracy"; review a sample of disagreements by hand |
| Free-tier token limits | Stratified sample, caching, cheap-model-first routing |
| Vaccine-only narratives | State scope clearly; use FAERS structured data for drug-side signal comparison |
| Over-building the agent | Ship baselines and the triage step first; add steps only when the earlier one is measured |
| Ethical concerns about safety data | No causal claims; no product-safety conclusions; follow terms of use |

## 10. Do not claim

- That any vaccine or drug causes any event. VAERS cannot establish causality.
- Clinical or regulatory validation.
- That this replaces PV professionals. It is a decision-support and triage study.

## 11. Stretch ideas

- MedDRA-style hierarchy roll-ups (if you can obtain a free academic MedDRA licence; check MSSO terms).
- Arabic intake: translate a sample, measure the accuracy drop, and test a bilingual prompt (only if you can check Arabic quality).
- Use the MCP tools from project H for label lookup.
- Literature screening add-on (PubMed abstracts mentioning a product + event).

## 12. Open questions to decide together

1. Which report years and how large a sample?
2. Which LLMs: Groq only, Groq + Gemini, or local?
3. Human review UI: Streamlit or Next.js (you know both)?
4. Which robustness perturbations matter most?

## 13. Resume/DM hook (fill in only after measuring)

> Built a pharmacovigilance case-intake agent on [N] real VAERS reports: seriousness recall [x]% at [y]% review workload, event-coding F1 [x] vs [y] baseline, and showed agent coding changed [N] of the top-50 disproportionality signals.
