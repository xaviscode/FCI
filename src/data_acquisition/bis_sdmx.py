from __future__ import annotations
import pandas as pd
import requests
from typing import Optional


class BisSdmxClient:
    """
    Minimal BIS SDMX client for a SINGLE series key request.

    Base:
      https://stats.bis.org/api/v1/data/{FLOW}/{KEY}
    """
    def __init__(self, base_url: str = "https://stats.bis.org/api/v1", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch_series(self, flow: str, key: str) -> pd.DataFrame:
        url = f"{self.base_url}/data/{flow}/{key}"
        params = {"format": "sdmx-json"}
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()

        time_vals = j["structure"]["dimensions"]["observation"][0]["values"]
        time_ids = [v.get("id") or v.get("name") for v in time_vals]

        ds0 = j["dataSets"][0]

        if "series" in ds0 and ds0["series"]:
            first_series = next(iter(ds0["series"].values()))
            obs = first_series.get("observations", {})
        else:
            obs = ds0.get("observations", {})

        rows = []
        for k, v in obs.items():
            ti = int(str(k).split(":")[0])
            val = v[0] if isinstance(v, list) else v
            rows.append((time_ids[ti], val))

        df = pd.DataFrame(rows, columns=["date", "value"])

        df["date"] = pd.PeriodIndex(df["date"], freq="Q").to_timestamp("Q")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        return df