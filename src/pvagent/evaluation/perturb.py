"""Synthetic noise applied to REAL narratives for robustness tests. Outputs are labelled as perturbed."""
from __future__ import annotations

import random
import re

ABBREV = {
    "patient": "pt", "hospital": "hosp", "history": "hx", "with": "w/", "without": "w/o", "shortness of breath": "SOB",
    "emergency room": "ER", "emergency department": "ED", "blood pressure": "BP", "heart rate": "HR",
    "approximately": "approx", "symptoms": "sx", "treatment": "tx", "diagnosis": "dx", "prescription": "rx",
    "vaccine": "vax", "vaccination": "vax", "injection": "inj", "headache": "HA", "nausea and vomiting": "N/V",
    "chest pain": "CP", "minutes": "mins", "hours": "hrs", "days": "d", "left": "L", "right": "R", "fever": "fvr",
}
LAY = {
    "pyrexia": "a really high temperature", "fever": "felt super hot", "dyspnea": "couldn't catch my breath",
    "dyspnoea": "couldn't catch my breath", "shortness of breath": "couldn't breathe right",
    "syncope": "passed out", "fainted": "blacked out", "erythema": "redness", "urticaria": "hives",
    "pruritus": "itching", "myalgia": "aching muscles", "arthralgia": "sore joints", "fatigue": "wiped out",
    "headache": "pounding head", "nausea": "felt sick to my stomach", "vomiting": "threw up",
    "diarrhea": "the runs", "tachycardia": "heart racing", "palpitations": "heart pounding",
    "lymphadenopathy": "swollen glands", "paresthesia": "pins and needles", "dizziness": "lightheaded",
    "rash": "spots on skin", "chills": "shivering", "malaise": "felt off", "swelling": "puffiness",
}
KEYBOARD = "qwertyuiopasdfghjklzxcvbnm"


def _sub_dict(text: str, mapping: dict[str, str]) -> str:
    for k in sorted(mapping, key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(k) + r"\b", mapping[k], text, flags=re.I)
    return text


def typos(text: str, rate: float = 0.08, seed: int = 0) -> str:
    rng = random.Random(seed)
    out = []
    for w in text.split(" "):
        if len(w) > 3 and rng.random() < rate:
            i = rng.randrange(len(w) - 1)
            op = rng.choice(["swap", "drop", "sub"])
            if op == "swap":
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif op == "drop":
                w = w[:i] + w[i + 1:]
            else:
                w = w[:i] + rng.choice(KEYBOARD) + w[i + 1:]
        out.append(w)
    return " ".join(out)


def abbreviations(text: str, seed: int = 0) -> str:
    return _sub_dict(text, ABBREV)


def lay_language(text: str, seed: int = 0) -> str:
    return _sub_dict(text, LAY)


def messy(text: str, seed: int = 0) -> str:
    """All lower case, no punctuation, plus light typos: a hurried phone-intake transcript."""
    return typos(re.sub(r"[^\w\s/]", " ", text.lower()), rate=0.04, seed=seed)


PERTURBATIONS = {"typos": typos, "abbreviations": abbreviations, "lay_language": lay_language, "messy": messy}


def translation_round_trip(client, text: str, lang: str = "Arabic") -> str:
    """Optional: translate to `lang` and back with the LLM (costs 2 calls per case; cached)."""
    from ..agent.llm import parse_json
    msgs = [{"role": "user", "content": f'Translate to {lang}. Reply as JSON {{"t": "..."}}.\n\n{text[:3000]}'}]
    fwd = parse_json(client.chat(msgs).text)["t"]
    back = [{"role": "user", "content": f'Translate to English. Reply as JSON {{"t": "..."}}.\n\n{fwd}'}]
    return parse_json(client.chat(back).text)["t"]
