# Data terms checklist (Phase 0)

Fill this in after reading the source documents yourself. Items marked "verify" are my understanding, not checked.

## VAERS
- Data Use Guide: https://vaers.hhs.gov/docs/VAERSDataUseGuide_en_September2021.pdf (check for a newer version on
  https://vaers.hhs.gov/data/dataguide.html)
- [ ] Read the guide's limitations section. Key points to quote in REPORT.md (verify wording):
  - Reports are unverified and anyone can submit; a report does not mean the vaccine caused the event.
  - VAERS cannot be used to calculate incidence or rates of adverse events.
  - Under-reporting, stimulated reporting (e.g. media attention) and duplicate reports occur.
- [ ] Symptom terms are coded by VAERS staff (MedDRA, per the guide; verify). Use terms only as they appear in the
  data. Do not redistribute any MedDRA files or hierarchy.
- [ ] Non-domestic reports have the free-text narrative removed (verify), so only domestic files are used.
- [ ] Confirm whether there are any restrictions on redistributing the CSVs. Until confirmed: do not commit raw data
  (`data/` is git-ignored) and do not publish individual narratives in the demo; show at most short excerpts.
- [ ] Confirm the field names used in `src/pvagent/data/load.py` against the guide's codebook
  (DIED, L_THREAT, ER_VISIT, HOSPITAL, X_STAY, DISABLE, BIRTH_DEFECT, OFC_VISIT, ER_ED_VISIT, SYMPTOM1..5).

## DailyMed
- Public NLM web services: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
- Labels fetched are the *current* SPL version, not the version in force at the report date.

## LLM providers
- Narratives are public data, but check your provider's data-retention terms before sending them.
- Groq free-tier limits (requests/day, tokens/day per model): check https://console.groq.com/settings/limits.

## Decisions (ideation section 12) — defaults chosen, change in config/config.yaml
| Question | Default | Why |
|---|---|---|
| Years / sample | 2023 + 2024; 5,000 stratified (50% serious); LLM on 600 test cases | Post-peak COVID volumes are manageable; recent label versions |
| LLMs | Groq gpt-oss-20b (small) + gpt-oss-120b (large); Gemini/Ollama optional | Same models as project H; free tier |
| Review UI | Streamlit | Fastest to ship; free hosting on Streamlit Community Cloud |
| Robustness | typos, abbreviations, lay language, messy transcript | Cover common intake noise; translation is a stretch |
