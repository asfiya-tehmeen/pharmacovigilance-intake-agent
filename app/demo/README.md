# Demo set

29 cases from the main test run (`results/20261128-114738-378036_agent_small_retrieval`, gpt-oss-20b, prompt intake-v2),
chosen to mix serious / non-serious and auto-processed / routed cases, plus one serious case that every triage layer
missed (its seriousness came only from the report form).

- `cases.jsonl`: the agent's structured output, including the VAERS narrative it read.
- `official.json`: the official VAERS coding for the same reports (seriousness flags, coded terms, vaccine type), shown
  in the app only for comparison.

Source: VAERS public data (vaers.hhs.gov), domestic reports 2023-2024. VAERS reports are unverified and cannot establish
that a vaccine caused an event.
