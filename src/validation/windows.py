from __future__ import annotations

from typing import List, Dict
import pandas as pd


def get_baseline_stress_windows() -> List[Dict[str, str]]:
    """
    Baseline narrative stress windows from WP1.
    Dates are inclusive and will be interpreted at monthly frequency.
    """
    return [
        {"episode": "Dot-com bust", "start": "2000-01-01", "end": "2002-12-31"},
        {"episode": "Global Financial Crisis", "start": "2007-07-01", "end": "2009-06-30"},
        {"episode": "Euro-area sovereign stress", "start": "2010-01-01", "end": "2012-12-31"},
        {"episode": "China/commodity/EM stress", "start": "2015-06-01", "end": "2016-03-31"},
        {"episode": "COVID-19 shock", "start": "2020-02-01", "end": "2020-06-30"},
        {"episode": "Inflation/tightening stress", "start": "2022-01-01", "end": "2023-12-31"},
    ]


def label_windows(
    df: pd.DataFrame,
    date_col: str,
    country_col: str,
    value_col: str,
    stress_windows: List[Dict[str, str]],
    transition_buffer_months: int = 3,
) -> pd.DataFrame:
    """
    Label each observation as:
      - 'stress'
      - 'buffer'
      - 'tranquil'

    Buffer = +/- N months around stress windows, excluding stress itself.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["window_label"] = "tranquil"
    out["episode"] = ""

    for w in stress_windows:
        start = pd.to_datetime(w["start"])
        end = pd.to_datetime(w["end"])
        mask = (out[date_col] >= start) & (out[date_col] <= end)
        out.loc[mask, "window_label"] = "stress"
        out.loc[mask, "episode"] = w["episode"]

    for w in stress_windows:
        start = pd.to_datetime(w["start"])
        end = pd.to_datetime(w["end"])
        b_start = start - pd.DateOffset(months=transition_buffer_months)
        b_end = end + pd.DateOffset(months=transition_buffer_months)

        mask = (out[date_col] >= b_start) & (out[date_col] <= b_end)
        buffer_only = mask & (out["window_label"] == "tranquil")
        out.loc[buffer_only, "window_label"] = "buffer"

    cols = [date_col, country_col, value_col, "window_label", "episode"]
    return out[cols].sort_values([country_col, date_col]).reset_index(drop=True)