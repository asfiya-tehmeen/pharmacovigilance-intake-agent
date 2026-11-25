"""Load VAERS yearly CSVs into DuckDB.

Expected input in raw_dir, per year, either:
  {YEAR}VAERSData.zip                      (as downloaded from vaers.hhs.gov/data/datasets.html)
or the extracted files:
  {YEAR}VAERSDATA.csv, {YEAR}VAERSSYMPTOMS.csv, {YEAR}VAERSVAX.csv

VAERS CSVs are latin-1 encoded; flag columns are 'Y' or blank.
Only domestic files are used: non-domestic reports have their free text removed.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

FLAG_COLS = ["DIED", "L_THREAT", "ER_VISIT", "HOSPITAL", "X_STAY", "DISABLE", "BIRTH_DEFECT", "OFC_VISIT", "ER_ED_VISIT"]
# ICH E2A seriousness criteria available in VAERS. ER visit alone is not "serious" under ICH.
SERIOUS_COLS = ["DIED", "L_THREAT", "HOSPITAL", "X_STAY", "DISABLE", "BIRTH_DEFECT"]
TABLES = ("VAERSDATA", "VAERSSYMPTOMS", "VAERSVAX")


def _read_year(raw_dir: Path, year: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    zpath = raw_dir / f"{year}VAERSData.zip"
    zf = zipfile.ZipFile(zpath) if zpath.exists() else None
    for t in TABLES:
        name = f"{year}{t}.csv"
        if zf is not None:
            member = next((m for m in zf.namelist() if Path(m).name.upper() == name.upper()), None)
            if member is None:
                raise FileNotFoundError(f"{name} not in {zpath}")
            raw = zf.read(member)
            df = pd.read_csv(io.BytesIO(raw), dtype=str, encoding="latin-1", keep_default_na=False)
        else:
            p = raw_dir / name
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing {p}. Download {year}VAERSData.zip from https://vaers.hhs.gov/data/datasets.html "
                    f"into {raw_dir}."
                )
            df = pd.read_csv(p, dtype=str, encoding="latin-1", keep_default_na=False)
        df.columns = [c.strip().upper() for c in df.columns]
        out[t] = df
    return out


def _clean_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.copy()
    for c in FLAG_COLS:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].str.strip().str.upper().eq("Y")
    df["SERIOUS"] = df[SERIOUS_COLS].any(axis=1)
    df["REPORT_YEAR"] = year
    df["VAERS_ID"] = pd.to_numeric(df["VAERS_ID"], errors="coerce").astype("Int64")
    df["AGE_YRS"] = pd.to_numeric(df.get("AGE_YRS"), errors="coerce")
    df["NUMDAYS"] = pd.to_numeric(df.get("NUMDAYS"), errors="coerce")
    df["HOSPDAYS"] = pd.to_numeric(df.get("HOSPDAYS"), errors="coerce")
    for c in ("RECVDATE", "VAX_DATE", "ONSET_DATE", "DATEDIED"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%m/%d/%Y", errors="coerce")
    df["SYMPTOM_TEXT"] = df["SYMPTOM_TEXT"].fillna("").str.strip()
    keep = ["VAERS_ID", "REPORT_YEAR", "RECVDATE", "STATE", "AGE_YRS", "SEX", "SYMPTOM_TEXT", *FLAG_COLS,
            "SERIOUS", "HOSPDAYS", "RECOVD", "VAX_DATE", "ONSET_DATE", "NUMDAYS", "DATEDIED", "OTHER_MEDS",
            "CUR_ILL", "HISTORY", "ALLERGIES", "V_ADMINBY", "SPLTTYPE", "FORM_VERS"]
    for c in keep:
        if c not in df.columns:
            df[c] = None
    return df[keep]


def _symptoms_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(1, 6):
        s, v = f"SYMPTOM{i}", f"SYMPTOMVERSION{i}"
        if s not in df.columns:
            continue
        part = df[["VAERS_ID", s]].rename(columns={s: "TERM"})
        part["VERSION"] = df[v] if v in df.columns else None
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    long["TERM"] = long["TERM"].str.strip()
    long = long[long["TERM"] != ""]
    long["VAERS_ID"] = pd.to_numeric(long["VAERS_ID"], errors="coerce").astype("Int64")
    return long.drop_duplicates(["VAERS_ID", "TERM"])


def _clean_vax(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["VAERS_ID"] = pd.to_numeric(df["VAERS_ID"], errors="coerce").astype("Int64")
    for c in ("VAX_TYPE", "VAX_MANU", "VAX_LOT", "VAX_DOSE_SERIES", "VAX_ROUTE", "VAX_SITE", "VAX_NAME"):
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].str.strip()
    return df[["VAERS_ID", "VAX_TYPE", "VAX_MANU", "VAX_LOT", "VAX_DOSE_SERIES", "VAX_ROUTE", "VAX_SITE", "VAX_NAME"]]


def load_vaers(raw_dir: Path, years: list[int], db_path: Path) -> dict:
    """Load years into DuckDB tables reports / symptoms / vax. Returns a data-quality summary."""
    data, sym, vax = [], [], []
    for y in years:
        t = _read_year(raw_dir, y)
        data.append(_clean_data(t["VAERSDATA"], y))
        sym.append(_symptoms_long(t["VAERSSYMPTOMS"]))
        vax.append(_clean_vax(t["VAERSVAX"]))
    reports = pd.concat(data, ignore_index=True)
    symptoms = pd.concat(sym, ignore_index=True)
    vaxes = pd.concat(vax, ignore_index=True)

    n_raw = len(reports)
    dup_ids = int(reports["VAERS_ID"].duplicated().sum())
    reports = reports.drop_duplicates("VAERS_ID", keep="last")
    # Possible duplicate reports of the same case: identical narrative + age + sex + vax date.
    key = pd.util.hash_pandas_object(reports[["SYMPTOM_TEXT", "AGE_YRS", "SEX", "VAX_DATE"]], index=False)
    nonempty = reports["SYMPTOM_TEXT"].str.len() > 0
    reports["POSSIBLE_DUP"] = nonempty & key.duplicated(keep="first")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    for name, df in (("reports", reports), ("symptoms", symptoms), ("vax", vaxes)):
        con.register("_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")
    summary = {
        "years": years,
        "reports_raw": n_raw,
        "duplicate_ids_dropped": dup_ids,
        "reports": len(reports),
        "possible_duplicate_cases": int(reports["POSSIBLE_DUP"].sum()),
        "missing_narrative": int((~nonempty).sum()),
        "serious": int(reports["SERIOUS"].sum()),
        "symptom_rows": len(symptoms),
        "distinct_terms": int(symptoms["TERM"].nunique()),
        "vax_rows": len(vaxes),
        "reports_without_symptoms": int((~reports["VAERS_ID"].isin(symptoms["VAERS_ID"])).sum()),
    }
    con.close()
    return summary
