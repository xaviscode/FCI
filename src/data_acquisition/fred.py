from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import requests

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"


class FredClient:
    """
    Minimal FRED client using the official series observations endpoint.

    - Reads API key from env var FRED_API_KEY by default.
    - Returns df with columns: ['date','value'] and numeric 'value'.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing FRED API key")
        self.timeout = timeout

    def fetch_series(self, series_id: str) -> pd.DataFrame:
        params = {"series_id": series_id, "api_key": self.api_key, "file_type": "json"}
        r = requests.get(FRED_API_BASE, params=params, timeout=self.timeout)
        r.raise_for_status()

        data = r.json().get("observations", [])
        df = pd.DataFrame(data)[["date", "value"]]

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])

        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df