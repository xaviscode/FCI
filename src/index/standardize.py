from __future__ import annotations

import pandas as pd


def apply_direction_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align all indicators so that higher = more stress/risk.

    direction = +1  => keep sign
    direction = -1  => multiply by -1
    """
    out = df.copy()
    out["value_aligned"] = out["value"] * out["direction"]
    return out


def winsorize_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    out_col: str = "value_winsor",
) -> pd.DataFrame:
    """
    Winsorize a value column within groups.
    """
    out = df.copy()

    def _winsorize(s: pd.Series) -> pd.Series:
        lo = s.quantile(lower_q)
        hi = s.quantile(upper_q)
        return s.clip(lower=lo, upper=hi)

    out[out_col] = out.groupby(group_cols)[value_col].transform(_winsorize)
    return out


def zscore_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    out_col: str = "z_value",
) -> pd.DataFrame:
    """
    Full-sample z-score within groups.
    """
    out = df.copy()

    def _z(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.Series([0.0] * len(s), index=s.index)
        return (s - s.mean()) / std

    out[out_col] = out.groupby(group_cols)[value_col].transform(_z)
    return out