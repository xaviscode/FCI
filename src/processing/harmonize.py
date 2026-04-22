from __future__ import annotations

import pandas as pd


def aggregate_to_monthly(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """
    Aggregate a daily/weekly/monthly series to month-end frequency.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")

    out["month"] = out["date"].dt.to_period("M").dt.to_timestamp("M")
    if how == "mean":
        g = out.groupby("month", as_index=False)["value"].mean()
    elif how == "last":
        g = out.groupby("month", as_index=False)["value"].last()
    else:
        raise ValueError(f"Unknown monthly aggregation: {how}")

    g = g.rename(columns={"month": "date"})
    g["date"] = pd.to_datetime(g["date"])
    return g


def normalize_quarterly_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force quarterly timestamps onto quarter-end.
    This prevents mismatches when computing missingness with pd.date_range(freq='Q').
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.to_period("Q").dt.to_timestamp("Q")
    return out.sort_values("date").reset_index(drop=True)


def quarterly_to_monthly_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quarterly -> Monthly step function:
      each quarter's value is copied to all months within that quarter.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.to_period("Q").dt.to_timestamp("Q")
    out = out.sort_values("date")

    rows = []
    for _, r in out.iterrows():
        q = pd.Period(r["date"], freq="Q")
        months = pd.period_range(q.start_time.to_period("M"), q.end_time.to_period("M"), freq="M")
        for m in months:
            rows.append((m.to_timestamp("M"), r["value"]))

    mdf = pd.DataFrame(rows, columns=["date", "value"]).sort_values("date").reset_index(drop=True)
    return mdf


def compute_missingness_by_freq(df: pd.DataFrame, freq: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """
    Compute missingness on a calendar matching freq:
      freq="M" -> month-end timeline
      freq="Q" -> quarter-end timeline
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")

    if freq == "Q":
        full = pd.date_range(start=start, end=end, freq="QE")
    elif freq == "M":
        full = pd.date_range(start=start, end=end, freq="ME")
    else:
        raise ValueError(f"Unsupported freq for missingness: {freq}")

    s = out.set_index("date")["value"].reindex(full)
    return float(s.isna().mean())


def term_spread(df_10y: pd.DataFrame, df_3m: pd.DataFrame) -> pd.DataFrame:
    """
    Term spread = 10Y - 3M (both monthly).
    """
    a = df_10y.set_index("date")["value"]
    b = df_3m.set_index("date")["value"]
    idx = a.index.union(b.index)
    out = (a.reindex(idx) - b.reindex(idx)).to_frame("value").reset_index().rename(columns={"index": "date"})
    return out.dropna().sort_values("date").reset_index(drop=True)


def yoy_pct_quarterly(df_q: pd.DataFrame, periods_q: int = 4) -> pd.DataFrame:
    """
    Quarterly YoY % change: (x / x_{t-4} - 1) * 100.
    """
    out = df_q.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.to_period("Q").dt.to_timestamp("Q")
    out = out.sort_values("date")
    out["value"] = (out["value"] / out["value"].shift(periods_q) - 1.0) * 100.0
    return out.dropna().reset_index(drop=True)


def yoy_pct_monthly(df_m: pd.DataFrame, periods_m: int = 12) -> pd.DataFrame:
    """
    Monthly YoY % change: (x / x_{t-12} - 1) * 100.
    """
    out = df_m.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp("M")
    out = out.sort_values("date")
    out["value"] = (out["value"] / out["value"].shift(periods_m) - 1.0) * 100.0
    return out.dropna().reset_index(drop=True)