# Pharmacovigilance Case-Intake Agent

Turns free-text vaccine adverse-event reports (real VAERS narratives) into structured, coded safety cases, routes
uncertain ones to a human, and measures whether its coding errors would change which disproportionality signals
(PRR/ROR) a safety team detects. Plan: [`ideation.md`](ideation.md).

**Live demo (review queue, 29 real VAERS test cases):** https://pharmacovigilance-intake-agent.streamlit.app/

> Decision-support and triage study. No causal claims: VAERS cannot establish that a vaccine causes an event.
> Not clinically or regulatorily validated. Does not replace PV professionals.

## What you need to provide
| Item | Where | Required? |
|---|---|---|
| **Groq API key** | https://console.groq.com/keys → put in `.env` as `GROQ_API_KEY=` | Yes (for the LLM steps) |
| **VAERS data** `2023VAERSData.zip`, `2024VAERSData.zip` | https://vaers.hhs.gov/data/datasets.html (CAPTCHA, so it must be a manual download) → `data/raw/` | Yes |
| Gemini API key | https://aistudio.google.com/apikey → `GEMINI_API_KEY=` | Optional backup provider |
| Ollama | https://ollama.com, `ollama pull llama3.1:8b` | Optional local provider |

Download the **domestic** yearly zips (not "NonDomestic", which has the narratives removed). Leave them zipped.

## Setup
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[app,dev]"
copy .env.example .env          # then paste your GROQ_API_KEY
pv check                        # verifies keys + data files
python -m pytest                # 16 offline tests (synthetic fixture + fake LLM; no keys needed)
```

## Run order
```powershell
pv load                 # VAERS CSVs -> DuckDB (reports / symptoms / vax), dedupe + data-quality summary
pv sample               # stratified 5,000-case sample, FROZEN dev/test split -> data/processed/sample_split.csv
pv labels --top 30      # DailyMed Adverse Reactions sections for the 30 most-reported vaccines
pv baselines            # rules + TF-IDF/LR triage, dictionary + nearest-term coding, product matcher (free)

pv agent --limit 20     # smoke test first: check tokens/case in the output, then size real runs
pv agent                # small model (gpt-oss-20b), retrieval-constrained coding, 600 test cases
pv agent --mode prompt_only          # ablation: free-text coding, no candidate list
pv agent --model large               # gpt-oss-120b
pv agent --hybrid                    # small first, escalate low-confidence cases to large
pv agent --limit 150 --consistency-k 2   # calibration experiment (3x calls per case)
pv sweep                # routing threshold vs human workload vs missed serious cases (latest agent run)
pv signal --agent-run results/<run_dir>   # headline: signals gained/lost/changed vs official coding
pv robustness --limit 100            # clean vs typos / abbreviations / lay language / messy
pv review               # Streamlit human-review queue
pv budget               # tokens used today per model
```
Every run writes `results/<timestamp>_<name>/` with `run.json`, `cases.jsonl` (one structured case per line) and
`metrics.json`. Evaluate on `--split dev` while iterating on prompts; use `test` only for reported numbers.

## Free-tier budget
Groq free tier is roughly 200K tokens/day per model (check your console; limits change). The client stops cleanly at
`llm.daily_token_budget` in `config/config.yaml` and **every response is cached**, so rerunning the same command the
next day resumes where it stopped at no cost. Tokens per case are only known after the smoke test; if it is ~2K,
the 600-case run takes ~6 days on one model. Options: lower `sample.llm_subset`, run small and large models on
different days in parallel (separate per-model limits), or add Gemini/Ollama.

## Design (maps to ideation section 5)
```
VAERS zips ─► load + dedupe ─► DuckDB ─► frozen stratified sample
                                              │
  1 rule triage (deterministic, recall-first; always runs)
  2 candidate retrieval: narrative clauses -> top-25 coded terms (char n-gram TF-IDF + fuzzy rerank)
  3 ONE LLM call: patient, reporter, products, events (+ term ids from the candidates), seriousness, outcome,
    each with a verbatim evidence quote
  4 span verifier: drops any field whose quote is not literally in the narrative; out-of-list term ids rejected
  5 validity: ICH four minimum elements
  6 expectedness: coded term vs DailyMed Adverse Reactions section (deterministic match)
  7 confidence = self-rating + span pass rate + rule/LLM triage agreement (+ self-consistency if enabled)
    route to human if low confidence, triage disagreement, invalid ICSR or an uncoded event
  8 signal layer: PRR/ROR (Evans criteria) on official coding vs baseline coders vs agent error model
```
Choices worth knowing:
- **One call per case** (extraction + coding together) instead of separate agent steps, to fit the free tier.
- **Serious = LLM OR rules** for triage, so the deterministic layer can only add recall, never remove it.
- **Signal impact at full scale** uses the agent's *measured* per-term recall and false-positive distribution
  (from the evaluated sample) applied to all reports, averaged over 5 seeds, because the LLM cannot be run on every
  report for free. A direct (small, noisy) comparison on the processed sample is reported too, plus an error-level
  sweep (recall × false positives per case) to show how much coding error the signal list tolerates.
  Products are kept as officially coded so the comparison isolates event-coding error.
- **Ground-truth caveat:** VAERS coded terms are assigned by VAERS staff from the full report, not only the
  narrative. Report this as *agreement with official coding*, not accuracy (ideation risk table).
- The **reporter** element is satisfied by the VAERS submission channel (identity is redacted in public data).
- Workload numbers are also given at the **natural serious-case rate**, because the sample oversamples serious cases.

## Layout
```
config/config.yaml        years, sample sizes, models, thresholds
src/pvagent/data/         load.py (VAERS -> DuckDB), sample.py (frozen split), labels.py (DailyMed)
src/pvagent/baselines/    triage_rules.py, triage_lr.py, coding.py, retrieval.py, product.py
src/pvagent/agent/        llm.py (cached, budgeted client), schema.py (E2B(R3)-inspired), intake.py, verifier.py,
                          expectedness.py
src/pvagent/evaluation/   metrics.py, signal.py, perturb.py, experiments.py
app/review_app.py         Streamlit review queue (decisions -> results/<run>/reviews.jsonl)
tests/                    offline tests on a SYNTHETIC fixture (never used for results)
notes/data-terms.md       VAERS / DailyMed terms of use checklist
```

## Status
Run on real VAERS data (2023-2024). Main results on 300 held-out test cases, ablations, calibration, signal impact and
robustness are in [`REPORT.md`](REPORT.md). Still running: self-consistency calibration and one robustness condition.

## Review UI: local and public demo
- **Local:** `pv review` shows the bundled demo set plus every run in `results/`; decisions go to `<run>/reviews.jsonl`.
- **Public demo:** `app/demo/` holds 29 test cases (agent output plus the official VAERS coding for comparison). On
  Streamlit Community Cloud, or with `PV_DEMO_ONLY=1`, the app shows only this set and keeps decisions in the
  visitor's session (downloadable as JSON). No LLM calls, no API keys, no database.
- Full narratives in `results/*/cases.jsonl` are git-ignored; only the demo set is published.

Deploy on Streamlit Community Cloud:
1. Push this repo to GitHub.
2. At https://share.streamlit.io sign in with GitHub, choose **Create app → Deploy a public app from GitHub**.
3. Repository `asfiya-tehmeen/pharmacovigilance-intake-agent`, branch `main`, main file path `app/review_app.py`.
4. Advanced settings: Python 3.13 (the version the app was tested on). No secrets are needed. Dependencies come from `app/requirements.txt`.
5. Deploy. Live at https://pharmacovigilance-intake-agent.streamlit.app/

## Not done / stretch
MedDRA hierarchy roll-ups (licence needed), Arabic translation round-trip (`evaluation/perturb.py:
translation_round_trip` exists but is not wired into a command), FAERS drug-side signal comparison via openFDA,
label-version pinning by report date, Loom demo.
