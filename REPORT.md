# Pharmacovigilance case-intake agent: report (DRAFT, results pending)

**Status:** baselines measured on the frozen test split. The LLM agent has only been run on the *dev* split (for prompt
iteration). Every section marked **PENDING** has not been run yet and contains no numbers. Do not quote from this
file until those sections are filled from `results/`.

## Scope and limits (read first)
- Data: VAERS public reports, domestic files, report years 2023 and 2024 (158,521 unique reports after dropping duplicate
  IDs; 148,701 with a usable narrative and not flagged as a possible duplicate case).
- VAERS reports are unverified and voluntary. **Nothing here supports any claim that a vaccine causes an event**, and
  VAERS cannot give incidence rates. Signals are statistical flags on report counts, not safety conclusions.
- "Ground truth" is VAERS's own coding, which was assigned from the whole report and form, not only the narrative. Results
  are therefore **agreement with official coding**, not accuracy. Some official terms are not recoverable from the text.
- Decision-support and triage study. Not clinically or regulatorily validated; it does not replace PV professionals.
- The LLM was run on a stratified **subset of 300 test cases** because of free-tier token limits (about 1.75K tokens per
  case). Confidence intervals will be wide; they are reported with every LLM number.
- Expectedness uses the **current (2026) DailyMed label** for 32 of the 40 most-reported vaccine names; the label in force
  at report time may differ. 8 names (emergency-use or "no brand name") have no label, so they are reported as `no_label`.
- The evaluation sample oversamples serious cases (50% vs about 14% in the data). Workload figures are therefore also
  given at the natural serious-case rate.

## Data and split
| Item | Value |
|---|---|
| Reports loaded (2023, 2024) | 161,244 raw, 158,521 after duplicate-ID removal |
| Possible duplicate cases (same narrative, age, sex, vaccination date) | 2,408 (excluded from sample) |
| Distinct official coded terms | 8,535 |
| Evaluation sample | 5,000 (2,500 serious), frozen split: 3,000 dev, 2,000 test |
| Natural serious rate (usable narratives) | 14.2% |

Serious = any of: death, life-threatening, hospitalisation, prolonged stay, disability, birth defect (ICH E2A criteria
available in VAERS; ER visit alone is not counted).

## Baselines (measured, 2,000 test cases; `results/*_baselines`)
| Task | Method | Result |
|---|---|---|
| Seriousness | Keyword rules | recall 0.624, precision 0.843 |
| Seriousness | TF-IDF logistic regression, threshold set for 99% recall on dev | recall 0.985, precision 0.784, AUROC 0.979; flags 37% of cases at the natural rate |
| Event coding | Exact dictionary match | micro-F1 0.360 (P 0.453, R 0.298) |
| Event coding | Nearest-term (TF-IDF) | micro-F1 0.308 |
| Product (vaccine type) | Dictionary | micro-F1 0.452 (0.548 by vaccine family) |
| Retrieval | Candidate list (45 retrieved + 40 most frequent terms): share of official terms offered to the LLM | 0.556 (ceiling on coding recall for retrieval-constrained coding) |

## Agent results: **PENDING** (main test run, 300 cases)
To fill from `results/<main run>/metrics.json`:
- Seriousness recall (model alone, rules, logistic regression, combined) and human-review workload at the natural rate.
- Event-coding precision / recall / F1 vs official terms, and vs the two baselines on the same cases.
- Product extraction, validity rate, span-verifier rejections, expectedness counts, tokens and seconds per case.

## Calibration and routing: **PENDING**
Confidence = mean of self-rating, span-pass rate and triage agreement (plus self-consistency in one experiment).
Early dev observation only (n = 132, not a result): the self-rating alone did not separate correct from incorrect cases
(AUROC about 0.60). Report ECE, reliability curve, risk-coverage curve and the workload-vs-missed-serious sweep.

## Ablations: **PENDING** (same first 100 test cases as the main run)
Retrieval-constrained vs free-text coding; small vs large model; self-consistency. Dev-only prior (n = 96 paired cases,
not a result): `gpt-oss-20b` and `gpt-oss-120b` coding F1 were 0.532 and 0.530.

## Signal impact (headline question): **PENDING**
Method: official coding vs (a) dictionary coder on all reports, (b) nearest-term coder on all reports, (c) the agent's
**measured** per-term recall and false-positive rate applied to all reports (a simulation, not an LLM run on every report),
averaged over 5 seeds, plus a recall x false-positive sweep. Signal rule: at least 3 reports, PRR >= 2, chi-square >= 4.
Provisional pipeline check on dev data: 148,701 reports, 4,876 official signals. These provisional figures are not results.

## Robustness: **PENDING** for the agent
Baseline-only check (measured, 400 test cases, `results/*_robustness`; perturbations are synthetic noise on real text):
| Perturbation | Rules recall | Dictionary F1 |
|---|---|---|
| none | 0.650 | 0.364 |
| typos | 0.615 | 0.351 |
| abbreviations | 0.640 | 0.350 |
| lay language | 0.650 | 0.301 |
| messy transcript | 0.630 | 0.359 |

## Failures and lessons so far (kept in the report on purpose)
- Keyword rules alone miss 38% of serious cases because many seriousness flags are never stated in the narrative; the
  logistic regression layer is what makes recall-first triage work.
- First retrieval ceiling was 0.35 of official terms; adding the most frequent coded terms raised it to 0.56.
- Coding conventions in the official data (for example `Hospitalisation` is almost never a coded term) had to be learned
  from the data and written into the prompt.
- Engineering bugs found by checking, not by tests alone: candidate-list ordering was non-deterministic (broke response
  caching), and one patch silently moved a loop body out of the loop. Both are fixed and covered by tests.

## Do not claim
Causality for any vaccine or event; clinical or regulatory validation; replacement of PV professionals.
