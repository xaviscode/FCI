from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict

import pandas as pd

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS series_metadata(
  series_id TEXT PRIMARY KEY,
  indicator_code TEXT,
  country_code TEXT,
  source_name TEXT,
  source_series_code TEXT,
  frequency TEXT,
  units TEXT,
  direction INTEGER,
  transform_spec TEXT,
  start_date TEXT,
  end_date TEXT,
  missingness REAL,
  is_fallback INTEGER,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS observations(
  series_id TEXT,
  date TEXT,
  value REAL,
  PRIMARY KEY(series_id, date)
);
"""


def ensure_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def upsert_series_metadata(db_path: str, row: Dict[str, Any]) -> None:
    cols = list(row.keys())
    placeholders = ",".join(["?"] * len(cols))
    update_set = ",".join([f"{c}=excluded.{c}" for c in cols if c != "series_id"])
    sql = f"""
    INSERT INTO series_metadata ({",".join(cols)})
    VALUES ({placeholders})
    ON CONFLICT(series_id) DO UPDATE SET {update_set}
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(sql, [row[c] for c in cols])
        conn.commit()


def insert_observations(db_path: str, series_id: str, df: pd.DataFrame) -> None:
    """
    Insert observations (date,value) for a series_id.
    df must have columns ['date','value'] with date as ISO string (yyyy-mm-dd).
    """
    if df.empty:
        return
    rows = [(series_id, str(d), float(v)) for d, v in zip(df["date"], df["value"])]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO observations(series_id,date,value) VALUES (?,?,?)",
            rows,
        )
        conn.commit()