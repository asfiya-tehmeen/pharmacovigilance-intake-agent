"""Dictionary product extraction: map narrative mentions to VAERS VAX_TYPE codes."""
from __future__ import annotations

import re

import pandas as pd

# Lay/abbreviation keywords per common VAERS VAX_TYPE; brand tokens from VAX_NAME are added automatically.
GENERIC = {
    "COVID19": ["covid", "covid-19", "covid19", "coronavirus", "sars-cov-2", "pfizer", "moderna", "novavax"],
    "COVID19-2": ["bivalent"],
    "FLU4": ["flu", "influenza", "flu shot"], "FLU3": ["flu", "influenza", "flu shot"],
    "FLUN4": ["flumist", "nasal flu"],
    "VARZOS": ["shingrix", "shingles", "zoster"], "ZOSTER": ["zostavax"],
    "HPV9": ["hpv", "gardasil", "papilloma"], "PNC20": ["prevnar", "pneumococcal", "pcv20"],
    "PNC21": ["capvaxive", "pcv21"], "PNC15": ["vaxneuvance", "pcv15"],
    "PPV": ["pneumovax", "ppsv23"], "TDAP": ["tdap", "boostrix", "adacel"],
    "TD": ["tenivac"], "MMR": ["mmr", "measles", "mumps", "rubella"], "VARCEL": ["varicella", "chickenpox", "varivax"],
    "HEPB": ["hepatitis b", "hep b", "engerix", "heplisav", "recombivax"], "HEPA": ["hepatitis a", "hep a", "havrix", "vaqta"],
    "RSV": ["rsv", "arexvy", "abrysvo", "respiratory syncytial"], "MNQ": ["menactra", "menveo", "menquadfi"],
    "MENB": ["bexsero", "trumenba", "men b"], "DTAP": ["dtap", "daptacel", "infanrix"],
    "RAB": ["rabies", "imovax", "rabavert"], "TYP": ["typhoid", "typhim"], "YF": ["yellow fever", "yf-vax"],
    "SMALLMNK": ["mpox", "monkeypox", "jynneos"], "RV5": ["rotateq"], "RV1": ["rotarix"],
}
_STOP = {"BRAND", "SEASONAL", "UNKNOWN", "QUADRIVALENT", "TRIVALENT", "VACCINE", "FOREIGN", "HIGH", "DOSE",
         "NAME", "PEDIATRIC", "ADULT", "BIVALENT"}


class ProductMatcher:
    def __init__(self, product_vocab: pd.DataFrame):
        kws: dict[str, set[str]] = {k: set(v) for k, v in GENERIC.items()}
        for _, r in product_vocab.iterrows():
            vt = str(r["VAX_TYPE"])
            kws.setdefault(vt, set())
            for inner in re.findall(r"\(([^()]+)\)", str(r["VAX_NAME"])):
                for tok in re.split(r"[ /]", inner):
                    if len(tok) >= 5 and tok.upper() not in _STOP:
                        kws[vt].add(tok.lower())
        self.patterns = {vt: [re.compile(r"\b" + re.escape(k) + r"\b", re.I) for k in sorted(ks)]
                         for vt, ks in kws.items() if ks}
        self.prior = product_vocab.groupby("VAX_TYPE")["n"].sum().to_dict()

    def match(self, text: str) -> list[str]:
        hits = [vt for vt, pats in self.patterns.items() if any(p.search(text or "") for p in pats)]
        # families sharing keywords (FLU3/FLU4, COVID19/COVID19-2): keep the most prevalent member
        fam: dict[str, str] = {}
        for vt in hits:
            key = re.sub(r"[0-9X]*(-\d+)?$", "", vt) or vt
            if key not in fam or self.prior.get(vt, 0) > self.prior.get(fam[key], 0):
                fam[key] = vt
        return sorted(fam.values())
