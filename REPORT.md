# Pharmacovigilance case-intake agent: report

**Status (2026-11-28):** main results measured on the frozen test split. Still running: **self-consistency
calibration** (17 of 60 cases) and the **messy-transcript** robustness case (10 of 40). Both are marked PENDING.

## Findings first
1. **Coding agreement.** On 300 held-out VAERS reports, the agent (gpt-oss-20b, retrieval-constrained) agreed with the
   official coded terms at **micro-F1 0.516** (95% CI 0.481–0.552), against **0.339** for an exact dictionary match and
   **0.305** for nearest-term matching on the same cases. Paired per-case F1 gain over the dictionary: **+0.30**
   (CI +0.26 to +0.33). Recall is the weak side (0.45): about half of the official terms are not recovered from the text.
2. **The candidate list matters.** Forcing the model to choose from retrieved candidate terms beat free-text coding
   by **+0.093** per-case F1 (CI +0.030 to +0.155, same 100 cases, same model). Without the list, **29.7%** of the
   model's terms did not exist in the VAERS term list at all (for example "covid-19 positive test" instead of
   "SARS-CoV-2 test positive").
3. **A larger model did not help.** gpt-oss-120b was slightly *worse* than gpt-oss-20b on the same 100 cases
   (per-case F1 −0.039, CI −0.078 to −0.002) and used 5% more tokens.
4. **Seriousness triage works only because of the classical layer.** The LLM alone found 64% of serious cases and the
   keyword rules 66%. A TF-IDF logistic regression found **98.0%**. All three combined found **98.7%** (148 of 150;
   CI 96.6–100%), which means sending **44%** of reports to review at the real-world serious rate (vs 36% for the
   regression alone). The LLM's added value for triage was one extra serious case out of 150.
5. **Do not auto-process.** After calibration on the dev split, confidence was well calibrated (ECE 0.08) but only
   moderately discriminative (AUROC 0.70–0.76). Even the 50% most confident cases had a **31%** case error rate
   (seriousness wrong or coding F1 below 0.5). Routing two thirds of cases to a human brought the error rate among the
   rest to only 26%.
6. **Signal impact (headline).** Applying the agent's measured coding errors to all 148,701 reports kept **47%** of the
   4,876 official disproportionality signals, lost 53%, added 462 new ones, and changed **23 of the top 50**. The loss
   falls mostly on weak signals: **79%** of signals backed by 50 or more reports survived, but only 13% of those backed by
   3–4 reports. The dictionary coder did worse (kept 31%, changed 29 of the top 50).
7. **Vaccine identification is not an LLM task here.** The agent named the vaccine type less well than a dictionary
   (family F1 0.33 vs 0.53), mostly because it left the code blank, and 127 of 300 narratives never name the vaccine.
   The VAERS form field is the right source.
8. **No large robustness drop.** Under typos, abbreviations and lay language, agent coding F1 moved by −0.04 to −0.01
   on 40 paired cases (no change distinguishable from zero), while the dictionary lost up to 0.06 under lay language.
   The sample only rules out drops of about 0.1 F1 or more.

## Scope and limits (read before using any number)
- Data: VAERS public reports, domestic files, report years 2023 and 2024 (158,521 unique reports; 148,701 with a usable
  narrative and not flagged as a possible duplicate case).
- VAERS reports are unverified and voluntary. **Nothing here supports any claim that a vaccine causes an event**, and
  VAERS cannot give incidence rates. Signals are statistical flags on report counts, not safety conclusions.
- "Ground truth" is VAERS's own coding, assigned from the whole report and form, not only the narrative. Results are
  **agreement with official coding**, not accuracy. Some official terms (and some seriousness flags, see below) cannot be
  recovered from the text at all.
- Decision-support and triage study. Not clinically or regulatorily validated; it does not replace PV professionals.
- The LLM ran on a stratified **300-case test subset** (150 serious) because of free-tier token limits (about 1,725
  tokens and 17 seconds per case). Ablations use 100 of those cases. Intervals are 95% bootstrap intervals over cases.
- The evaluation sample oversamples serious cases (50% vs 14.2% in the data); workload figures are converted to the
  natural serious rate.
- Prompts were developed on the dev split only (150 cases). The coding conventions added to the prompt (for example,
  hospitalisation is almost never a coded term) were measured on the whole dataset's term counts, not on test labels.
- Expectedness uses the **current (2026) DailyMed labels** for 32 of the 40 most-reported vaccine names; the label in force
  at report time may differ, and 8 names (emergency-use or "no brand name") have no label.
- The signal-impact result is a **simulation**: the agent's per-term recall and false-positive rate, measured on 300
  test cases, applied to every report (5 seeds; the spread across seeds was small, top-50 overlap 25–28). The LLM was
  not run on all 148,701 reports.

## Data and split
| Item | Value |
|---|---|
| Reports loaded (2023, 2024) | 161,244 raw, 158,521 after duplicate-ID removal |
| Possible duplicate cases (same narrative, age, sex, vaccination date) | 2,408 (excluded from the sample) |
| Distinct official coded terms | 8,535 |
| Evaluation sample | 5,000 (2,500 serious), frozen split: 3,000 dev, 2,000 test |
| Natural serious rate (usable narratives) | 14.2% |

Serious = any of: death, life-threatening, hospitalisation, prolonged stay, disability, birth defect (ICH E2A
criteria available in VAERS; an ER visit alone is not counted).

## Baselines (2,000 test cases)
| Task | Method | Result |
|---|---|---|
| Seriousness | Keyword rules | recall 0.624, precision 0.843 |
| Seriousness | TF-IDF logistic regression, threshold for 99% recall on dev | recall 0.985, precision 0.784, AUROC 0.979 |
| Event coding | Exact dictionary match | micro-F1 0.360 (P 0.453, R 0.298) |
| Event coding | Nearest term (character TF-IDF) | micro-F1 0.308 |
| Vaccine type | Dictionary | micro-F1 0.452 (0.548 by vaccine family) |
| Retrieval | Share of official terms present in the candidate list shown to the LLM | 0.556 (upper bound on coding recall) |

## Agent: main run (300 test cases, gpt-oss-20b, prompt intake-v2)
**Event coding**

| Method (same 300 cases) | micro-F1 (95% CI) | Precision | Recall |
|---|---|---|---|
| Agent, retrieval-constrained | **0.516** (0.481–0.552) | 0.610 | 0.447 |
| Dictionary | 0.339 (0.305–0.374) | 0.440 | 0.275 |
| Nearest term | 0.305 (0.276–0.335) | 0.418 | 0.241 |

Exact match of the full term set: 16.3% of cases.

**Seriousness (150 serious of 300)**

| Layer | Recall (95% CI) | Precision | Missed |
|---|---|---|---|
| LLM | 0.640 (0.563–0.715) | 0.914 | 54 |
| Keyword rules | 0.660 (0.582–0.735) | 0.884 | 51 |
| Logistic regression | 0.980 (0.955–1.000) | 0.795 | 3 |
| LLM or rules | 0.773 (0.707–0.841) | 0.847 | 34 |
| LLM or rules or regression | **0.987** (0.966–1.000) | 0.740 | **2** |

Review workload at the natural serious rate: 35.6% (regression alone) and 43.7% (all three layers).

The two serious cases every layer missed are not detectable from the text. One narrative reads, in full, "L upper thigh-
15 x 10 cm area of redness, warmth, and tightness" (flagged hospitalised); the other describes a rash after a shingles
vaccine (flagged disability). Neither mentions a hospital stay or a disability; the flags came from the report form.
A production intake would read the form's seriousness checkboxes as well as the narrative.

**Other checks**
- Evidence spans: 27 of 300 cases had at least one field dropped because its quote was not in the narrative (35 fields).
- Valid individual case report (four minimum elements): 92% of cases.
- Expectedness of coded events against the label: 265 listed, 645 not listed, 200 no label available, 45 not checked.
  "Not listed" is overstated: label text uses different wording from coded terms, and the check is a string match.
- Cost: 1,725 tokens and 17 seconds per case on the Groq free tier.

## Ablations (same first 100 test cases of the main run)
| Comparison | Result |
|---|---|
| Candidate list vs free-text coding (gpt-oss-120b both) | F1 0.532 vs 0.431; paired per-case **+0.093** (CI +0.030 to +0.155); free text produced 29.7% off-vocabulary terms; the list costs about 34% more tokens (1,812 vs 1,353) |
| gpt-oss-120b vs gpt-oss-20b (candidate list both) | F1 0.532 vs 0.547; paired per-case **−0.039** (CI −0.078 to −0.002); 1,812 vs 1,724 tokens per case |
| Self-consistency confidence (k = 2) | **PENDING** (17 of 60 cases) |

## Calibration and routing (fit on 150 dev cases, evaluated on 300 test cases)
"Correct" case = seriousness agrees with the VAERS flags **and** event-coding F1 is at least 0.5. 45.7% of test cases
were not correct by this definition.

| Confidence score | ECE | AUROC | Error rate among the 50% most confident | among the 80% most confident |
|---|---|---|---|---|
| Hand-weighted (as built) | 0.309 | 0.692 | 0.307 | 0.375 |
| Calibrated, same 3 inputs | **0.080** | 0.699 | 0.307 | 0.375 |
| Calibrated, 9 case features | 0.131 | **0.757** | 0.320 | 0.354 |

Routing sweep on the calibrated score (cases with triage disagreement or an invalid report always go to a human):

| Calibrated threshold | Sent to human review | Error rate in auto-processed cases | Serious cases auto-processed as non-serious |
|---|---|---|---|
| none | 37.3% | 0.356 | 2 |
| 0.5 | 41.0% | 0.339 | 2 |
| 0.7 | 58.7% | 0.282 | 2 |
| 0.8 | 67.3% | 0.255 | 2 |

The as-built hand-weighted confidence was bunched above 0.8, so a threshold sweep on it barely changed routing.

## Signal impact (headline question)
Signal rule: at least 3 reports, PRR at least 2, chi-square at least 4, on (vaccine type, coded term) pairs over
148,701 reports. **4,876 official signals.**

| Coding used for the signal table | Signals | Kept | Lost | Gained | Jaccard | Top-50 unchanged |
|---|---|---|---|---|---|---|
| Official (reference) | 4,876 | — | — | — | 1.00 | 50 |
| Agent error model (mean of 5 seeds) | 2,736 | 2,274 | 2,602 | 462 | **0.43** | **27** |
| Dictionary coder, all reports | 2,914 | 1,509 | 3,367 | 1,405 | 0.24 | 21 |
| Nearest-term coder, all reports* | 3,504 | 1,505 | 3,371 | 1,999 | 0.22 | 24 |

Retention by strength of the official signal (agent error model, seed 0):

| Reports behind the official signal | Signals | Kept |
|---|---|---|
| 3–4 | 1,102 | 13% |
| 5–9 | 1,254 | 36% |
| 10–19 | 1,008 | 56% |
| 20–49 | 791 | 72% |
| 50 or more | 721 | **79%** |

For the 27 vaccine types with at least 1,000 reports, 74% of signals backed by 20 or more reports survived.

Error-level sweep (synthetic error, uniform across terms)*: at term recall 0.9 with 0.5 false terms per report the
signal list keeps Jaccard 0.77; at recall 0.5 with 1 false term per report, 0.41. The agent's measured operating point
(recall 0.45, 1.38 false terms per report) gives 0.43, consistent with the sweep.

\* Computed on 2026-11-25 on the same data; they do not depend on the agent run. Rerun with `pv signal --with-nearest`
when memory allows; they may shift slightly because term ordering was made deterministic afterwards.

What this means: an intake agent at this level of agreement with official coding would preserve most strong signals
but would silently drop many weak ones and invent a few hundred new ones. Coding recall, not precision, drives the loss.

## Robustness (synthetic noise applied to real narratives)
**Agent** (gpt-oss-120b, 40 test cases, 20 serious; paired against the same cases' clean text). The regression layer
was re-scored on the perturbed text; the keyword rules and the LLM saw the perturbed text directly.

| Perturbation | Agent coding F1 | Paired change vs clean (95% CI) | Dictionary F1, same cases | Seriousness recall: LLM / rules / regression / combined |
|---|---|---|---|---|
| none | 0.582 | — | 0.355 | 0.70 / 0.75 / 1.00 / 1.00 |
| typos | 0.559 | −0.037 (−0.111 to +0.022) | 0.339 | 0.65 / 0.75 / 1.00 / 1.00 |
| abbreviations | 0.548 | −0.022 (−0.094 to +0.049) | 0.324 | 0.65 / 0.75 / 1.00 / 1.00 |
| lay language | 0.571 | −0.008 (−0.027 to +0.009) | 0.291 | 0.65 / 0.75 / 1.00 / 1.00 |
| messy transcript | **PENDING** (10 of 40 cases) | | | |

No drop was distinguishable from zero. With 40 cases this only rules out large drops (roughly 0.1 F1 or more); it does
not show the agent is unaffected. Lay language hurt the dictionary most (−0.064) and the agent least.

**Baseline layers** at larger scale (400 test cases):

| Perturbation | Rules recall | Dictionary F1 |
|---|---|---|
| none | 0.650 | 0.364 |
| typos | 0.615 | 0.351 |
| abbreviations | 0.640 | 0.350 |
| lay language | 0.650 | 0.301 |
| messy transcript | 0.630 | 0.359 |


## Failures and lessons (kept on purpose)
- Keyword rules alone miss a third of serious cases because many seriousness flags are never written in the narrative.
- The first retrieval step offered only 35% of official terms; adding the 40 most frequent coded terms raised it to 56%.
- Coding conventions in the official data had to be learned and written into the prompt (prompt v2: coding F1 0.50 to
  0.53 on 96 paired dev cases, within noise).
- The LLM left the vaccine code blank for 49% of the products it extracted.
- Engineering bugs found by checking results rather than by tests alone: candidate-list order was not deterministic
  (silently broke response caching), one patch moved a loop body outside the loop, and a script edit corrupted a file path
  so that three experiments were skipped. In robustness runs the regression layer first scored the clean text instead
  of the perturbed one (corrected in the analysis above and in the code). All are fixed; the first two are covered by tests.

## Reproduce
See `README.md`. Main run: `results/20261128-114738-378036_agent_small_retrieval`; dev run:
`results/20261128-114724-527965_agent_small_retrieval`; signal: `results/20261128-132649-343784_signal`.

## Do not claim
Causality for any vaccine or event; clinical or regulatory validation; replacement of PV professionals.
