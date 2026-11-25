"""DailyMed label lookup for expectedness checks.

Pulls the Adverse Reactions section (LOINC 34084-4) of the current SPL for each product in scope.
Limitation: this is the *current* label version; the label in force at report time may differ.
DailyMed exposes history at /spls/{setid}/history if you need to pin versions.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import duckdb
import httpx

BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
NS = {"v3": "urn:hl7-org:v3"}
AR_CODES = {"34084-4"}  # ADVERSE REACTIONS SECTION

# VAERS VAX_NAME -> DailyMed search string, for names the heuristic gets wrong. Extend as needed.
OVERRIDES = {
    "COVID19 (COVID19 (PFIZER-BIONTECH))": "COMIRNATY",
    "COVID19 (COVID19 (MODERNA))": "SPIKEVAX",
    "COVID19 (COVID19 (NOVAVAX))": "NOVAVAX COVID-19",
    "RSV (AREXVY)": "AREXVY",
    "RSV (ABRYSVO)": "ABRYSVO",
    "PNEUMO (PREVNAR20)": "PREVNAR 20",
    "MEASLES + MUMPS + RUBELLA (MMR II)": "M-M-R II",
    "COVID19 (COVID19 (JANSSEN))": "JANSSEN COVID-19",
}


def search_term_for(vax_name: str) -> str:
    if vax_name in OVERRIDES:
        return OVERRIDES[vax_name]
    inner = re.findall(r"\(([^()]+)\)", vax_name)
    term = inner[-1] if inner else vax_name
    term = re.sub(r"\b(NO BRAND NAME|UNKNOWN|FOREIGN)\b", "", term, flags=re.I).strip(" -")
    return term or vax_name


def _section_text(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    chunks = []
    for sec in root.iter("{urn:hl7-org:v3}section"):
        code = sec.find("v3:code", NS)
        if code is not None and code.get("code") in AR_CODES:
            chunks.append(" ".join(t.strip() for t in sec.itertext() if t.strip()))
    return "\n".join(chunks)


def fetch_label(client: httpx.Client, query: str) -> dict | None:
    r = client.get(f"{BASE}/spls.json", params={"drug_name": query, "pagesize": 5})
    r.raise_for_status()
    hits = r.json().get("data", [])
    if not hits:
        return None
    best = hits[0]
    x = client.get(f"{BASE}/spls/{best['setid']}.xml")
    x.raise_for_status()
    return {"setid": best["setid"], "title": best.get("title", ""), "published_date": best.get("published_date", ""),
            "ar_text": _section_text(x.content)}


def query_variants(q: str) -> list[str]:
    """Full query, letter/digit split (PREVNAR13 -> PREVNAR 13), then progressively fewer leading words."""
    spaced = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", q)
    out = [q, spaced]
    words = spaced.split()
    out += [" ".join(words[:n]) for n in range(len(words) - 1, 0, -1)]
    seen, res = set(), []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            res.append(v)
    return res


# Names with no current SPL (emergency-use or generic placeholders): skip instead of matching something unrelated.
NO_LABEL = ("NO BRAND NAME", "UNKNOWN", "NOT SPECIFIED", "(OTHER)", "JANSSEN", "BIVALENT")


def fetch_labels(db_path, top_n: int = 30, sleep_s: float = 0.3) -> dict:
    """Fetch labels for the top_n most-reported VAX_NAMEs and cache them in table `labels`."""
    con = duckdb.connect(str(db_path))
    con.execute("""CREATE TABLE IF NOT EXISTS labels (vax_name VARCHAR PRIMARY KEY, query VARCHAR, setid VARCHAR,
                   title VARCHAR, published_date VARCHAR, ar_text VARCHAR, retrieved_at VARCHAR)""")
    names = [r[0] for r in con.execute(
        f"SELECT VAX_NAME FROM vax WHERE VAX_NAME <> '' GROUP BY 1 ORDER BY count(*) DESC LIMIT {int(top_n)}").fetchall()]
    have = {r[0] for r in con.execute("SELECT vax_name FROM labels WHERE length(ar_text) > 0").fetchall()}
    got, missing = 0, []
    with httpx.Client(timeout=30, headers={"User-Agent": "pvagent-research/0.1"}) as client:
        for name in names:
            if name in have:
                continue
            q = search_term_for(name)
            lab = None
            if not any(m in name.upper() for m in NO_LABEL):
                try:
                    for variant in query_variants(q):
                        lab = fetch_label(client, variant)
                        if lab and lab["ar_text"]:
                            q = variant
                            break
                except httpx.HTTPError as e:
                    missing.append((name, f"http error: {e}"))
                    continue
            if not lab or not lab["ar_text"]:
                missing.append((name, q))
                lab = lab or {"setid": None, "title": None, "published_date": None, "ar_text": ""}
            else:
                got += 1
            con.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?)",
                        [name, q, lab["setid"], lab["title"], lab["published_date"], lab["ar_text"],
                         datetime.now(timezone.utc).isoformat()])
            time.sleep(sleep_s)
    con.close()
    return {"requested": len(names), "fetched_with_ar_section": got, "missing": missing}
