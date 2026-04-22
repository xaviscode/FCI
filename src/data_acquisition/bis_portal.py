from __future__ import annotations

import csv
from io import StringIO

import pandas as pd
import requests


class BisPortalClient:
    """
    Download BIS data portal exports as "long CSV".

    We locate the actual data header (TIME_PERIOD / OBS_VALUE) and ignore leading metadata blocks.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def fetch_series(self, dataset: str, key: str) -> pd.DataFrame:
        base = "https://data.bis.org/topics"
        topic = "CREDIT_GAPS" if dataset == "WS_CREDIT_GAP" else "DSR"
        url = f"{base}/{topic}/BIS%2C{dataset}%2C1.0/{key}"
        params = {"file_format": "csv", "format": "long"}

        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        text = r.text
        lines = text.splitlines()

        sample = "\n".join(lines[:80])
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            sep = dialect.delimiter
        except Exception:
            sep = ","

        header_idx = None
        for i, line in enumerate(lines):
            l = line.lower()
            if "time_period" in l and "obs_value" in l:
                header_idx = i
                break
        if header_idx is None:
            raise ValueError("Could not locate BIS data header (TIME_PERIOD / OBS_VALUE).")

        data_text = "\n".join(lines[header_idx:])
        df = pd.read_csv(StringIO(data_text), sep=sep, engine="python", on_bad_lines="skip")

        def find_col(contains: str) -> str | None:
            contains = contains.lower()
            for c in df.columns:
                if contains in c.lower():
                    return c
            return None

        time_col = find_col("time_period")
        val_col = find_col("obs_value")

        if time_col is None or val_col is None:
            raise ValueError(f"Unexpected BIS CSV columns: {list(df.columns)}")

        out = df[[time_col, val_col]].rename(columns={time_col: "date", val_col: "value"})
        out["value"] = pd.to_numeric(out["value"], errors="coerce")

        out["date"] = pd.PeriodIndex(out["date"].astype(str), freq="Q").to_timestamp("Q")
        out = out.dropna().sort_values("date").reset_index(drop=True)

        return out