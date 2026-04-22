from __future__ import annotations

import pandas as pd


def compare_stress_vs_tranquil(
    df: pd.DataFrame,
    value_col: str,
    label_col: str = "window_label",
) -> pd.DataFrame:
    """
    Compare mean/median/std in stress vs tranquil periods by country.
    Buffer periods are excluded from the comparison.
    """
    x = df[df[label_col].isin(["stress", "tranquil"])].copy()

    grouped = (
        x.groupby(["country_code", label_col])[value_col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    wide = grouped.pivot(index="country_code", columns=label_col)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    if f"mean_stress" in wide.columns and f"mean_tranquil" in wide.columns:
        wide["mean_diff_stress_minus_tranquil"] = wide["mean_stress"] - wide["mean_tranquil"]

    if f"median_stress" in wide.columns and f"median_tranquil" in wide.columns:
        wide["median_diff_stress_minus_tranquil"] = wide["median_stress"] - wide["median_tranquil"]

    return wide.sort_values("country_code").reset_index(drop=True)


def summarize_episode_means(
    df: pd.DataFrame,
    value_col: str,
    stress_windows: list[dict],
) -> pd.DataFrame:
    """
    Compute per-country average index value inside each named stress window.
    """
    rows = []

    for country, g in df.groupby("country_code"):
        for w in stress_windows:
            start = pd.to_datetime(w["start"])
            end = pd.to_datetime(w["end"])
            sub = g[(g["date"] >= start) & (g["date"] <= end)].copy()
            if sub.empty:
                continue

            rows.append(
                {
                    "country_code": country,
                    "episode": w["episode"],
                    "start": start,
                    "end": end,
                    "n_obs": len(sub),
                    "mean_value": sub[value_col].mean(),
                    "median_value": sub[value_col].median(),
                    "max_value": sub[value_col].max(),
                }
            )

    return pd.DataFrame(rows).sort_values(["country_code", "start"]).reset_index(drop=True)


def extract_top_peaks(
    df: pd.DataFrame,
    value_col: str,
    top_n: int = 10,
    min_spacing_months: int = 6,
) -> pd.DataFrame:
    """
    Extract top peaks per country with a minimum spacing rule.
    Greedy selection from highest to lower values.
    """
    rows = []

    for country, g in df.groupby("country_code"):
        g = g.sort_values("date").reset_index(drop=True)

        candidates = g[["date", "country_code", value_col, "window_label", "episode"]].copy()
        candidates = candidates.sort_values(value_col, ascending=False).reset_index(drop=True)

        selected_dates = []

        for _, r in candidates.iterrows():
            dt = pd.to_datetime(r["date"])

            too_close = False
            for sdt in selected_dates:
                month_gap = abs((dt.year - sdt.year) * 12 + (dt.month - sdt.month))
                if month_gap < min_spacing_months:
                    too_close = True
                    break

            if too_close:
                continue

            selected_dates.append(dt)
            rows.append(
                {
                    "country_code": country,
                    "date": dt,
                    "peak_value": r[value_col],
                    "window_label": r.get("window_label", ""),
                    "episode": r.get("episode", ""),
                }
            )

            if len(selected_dates) >= top_n:
                break

    return pd.DataFrame(rows).sort_values(["country_code", "date"]).reset_index(drop=True)