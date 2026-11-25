"""Deterministic, recall-first seriousness triage.

Runs before (and independently of) the LLM so a serious case is never dropped silently.
A match inside a short negation window ("not hospitalized", "no ER visit") is kept but marked negated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CRITERIA: dict[str, list[str]] = {
    "death": [r"\bdied\b", r"\bdeath\b", r"\bdeceased\b", r"\bpassed away\b", r"\bexpired\b", r"\bfatal\w*\b",
              r"\bautopsy\b", r"\bcoroner\b", r"\bpronounced dead\b", r"\bdemise\b"],
    "life_threatening": [r"life[- ]threatening", r"\banaphyla\w+", r"\bcardiac arrest\b", r"\bcode blue\b",
                         r"\bintubat\w+", r"\bcpr\b", r"\bresuscitat\w+", r"\bepinephrine\b|\bepi[- ]?pen\b",
                         r"\brespiratory (failure|arrest)\b", r"\bicu\b|intensive care", r"\bstatus epilepticus\b",
                         r"\bventilator\b", r"\bseptic shock\b|\bshock\b"],
    "hospitalisation": [r"\bhospitali[sz]\w*", r"\badmitted\b|\badmission\b", r"\binpatient\b",
                        r"\bin the hospital\b|\bin hospital\b|\bto the hospital\b", r"\bdischarged\b",
                        r"\bobservation (unit|stay)\b", r"\bicu\b|intensive care"],
    "disability": [r"\bdisab\w+", r"\bparaly\w+", r"\bpermanent\w*\b", r"\bunable to (walk|work|move)\b",
                   r"\bguillain[- ]barr\w*\b|\bgbs\b", r"\bbell'?s pals\w+", r"\bwheelchair\b"],
    "birth_defect": [r"\bbirth defect\w*\b", r"\bcongenital\b", r"\bmalformation\w*\b", r"\bmiscarriage\b",
                     r"\bstillbirth\b|\bstillborn\b", r"\bspontaneous abortion\b"],
}
NEGATORS = re.compile(r"\b(no|not|denie[sd]|without|never|negative for|did not|didn't|wasn't|was not)\b", re.I)
_compiled = {k: [re.compile(p, re.I) for p in v] for k, v in CRITERIA.items()}


@dataclass
class RuleResult:
    serious: bool
    score: float                      # 1.0 positive match, 0.5 only negated matches, 0.0 none
    criteria: dict[str, list[str]] = field(default_factory=dict)  # criterion -> matched quotes
    negated: list[str] = field(default_factory=list)


def triage(text: str) -> RuleResult:
    text = text or ""
    crit: dict[str, list[str]] = {}
    negated: list[str] = []
    for name, pats in _compiled.items():
        for p in pats:
            for m in p.finditer(text):
                window = text[max(0, m.start() - 30):m.start()]
                quote = text[m.start():m.end()]
                if NEGATORS.search(window):
                    negated.append(quote)
                else:
                    crit.setdefault(name, []).append(quote)
    if crit:
        return RuleResult(True, 1.0, crit, negated)
    # Recall-first: negated-only mentions still go to a human (score 0.5).
    return RuleResult(bool(negated), 0.5 if negated else 0.0, crit, negated)
