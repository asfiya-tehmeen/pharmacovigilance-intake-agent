"""Stratified evaluation sample with a frozen dev/test split, plus case accessors."""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

MIN_NARRATIVE_CHARS = 30


def build_sample(db_path: Path, size: int, serious_fraction: float, test_fraction: float, seed: int,
                 freeze_path: Path) -> dict:
    """Sample cases with a usable narrative, oversampling serious cases.

    The split is written to freeze_path (CSV). If that file already exists it is reused unchanged,
    so the test split never moves once created.
    """
    con = duckdb.connect(str(db_path))
    if freeze_path.exists():
        frozen = pd.read_csv(freeze_path)
        con.register("_s", frozen)
        con.execute("CREATE OR REPLACE TABLE sample AS SELECT * FROM _s")
        con.close()
        return {"reused_frozen_split": str(freeze_path), **frozen["split"].value_counts().to_dict()}

    pool = con.execute(f"""
        SELECT r.VAERS_ID, r.SERIOUS FROM reports r
        WHERE length(r.SYMPTOM_TEXT) >= {MIN_NARRATIVE_CHARS} AND NOT r.POSSIBLE_DUP
          AND r.VAERS_ID IN (SELECT VAERS_ID FROM symptoms)
          AND r.VAERS_ID IN (SELECT VAERS_ID FROM vax)
    """).df()
    rng = np.random.default_rng(seed)
    n_ser = min(int(round(size * serious_fraction)), int(pool["SERIOUS"].sum()))
    n_non = min(size - n_ser, int((~pool["SERIOUS"]).sum()))
    ser = pool[pool["SERIOUS"]].sample(n=n_ser, random_state=seed)
    non = pool[~pool["SERIOUS"]].sample(n=n_non, random_state=seed)
    s = pd.concat([ser, non]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # stratified split
    s["split"] = "dev"
    for flag in (True, False):
        idx = s.index[s["SERIOUS"] == flag].to_numpy()
        test_idx = rng.choice(idx, size=int(round(len(idx) * test_fraction)), replace=False)
        s.loc[test_idx, "split"] = "test"
    s = s[["VAERS_ID", "SERIOUS", "split"]]
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(freeze_path, index=False)
    con.register("_s", s)
    con.execute("CREATE OR REPLACE TABLE sample AS SELECT * FROM _s")
    con.close()
    return {"pool": len(pool), "pool_serious_rate": float(pool["SERIOUS"].mean()), "sampled": len(s),
            **s["split"].value_counts().to_dict(), "serious_in_sample": int(s["SERIOUS"].sum())}


def get_cases(con: duckdb.DuckDBPyConnection, split: str | None = None, limit: int | None = None,
              ids: list[int] | None = None) -> pd.DataFrame:
    """One row per case with narrative, ground-truth flags, terms list and vaccines list."""
    where = []
    if split:
        where.append(f"s.split = '{split}'")
    if ids is not None:
        where.append(f"r.VAERS_ID IN ({','.join(str(int(i)) for i in ids) or 'NULL'})")
    w = ("WHERE " + " AND ".join(where)) if where else ""
    lim = f"LIMIT {int(limit)}" if limit else ""
    return con.execute(f"""
        SELECT r.*, s.split,
          (SELECT list(TERM ORDER BY TERM) FROM symptoms y WHERE y.VAERS_ID = r.VAERS_ID) AS terms,
          (SELECT list(VAX_TYPE) FROM vax v WHERE v.VAERS_ID = r.VAERS_ID) AS vax_types,
          (SELECT list(VAX_NAME) FROM vax v WHERE v.VAERS_ID = r.VAERS_ID) AS vax_names,
          (SELECT list(VAX_MANU) FROM vax v WHERE v.VAERS_ID = r.VAERS_ID) AS vax_manus
        FROM reports r JOIN sample s USING (VAERS_ID)
        {w} ORDER BY r.VAERS_ID {lim}
    """).df()


def term_vocabulary(con: duckdb.DuckDBPyConnection, min_freq: int) -> pd.DataFrame:
    """Coded-term vocabulary from dev-split cases plus all non-sample reports (never the test split)."""
    return con.execute(f"""
        SELECT TERM, count(*) AS n FROM symptoms
        WHERE VAERS_ID NOT IN (SELECT VAERS_ID FROM sample WHERE split = 'test')
        GROUP BY TERM HAVING count(*) >= {int(min_freq)} ORDER BY n DESC, TERM
    """).df()


def product_vocabulary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
        SELECT VAX_TYPE, VAX_NAME, VAX_MANU, count(*) AS n FROM vax
        GROUP BY ALL ORDER BY n DESC, VAX_TYPE, VAX_NAME, VAX_MANU
    """).df()
