from __future__ import annotations

import pandas as pd


def compare_ew_vs_pca_correlations(
    fci_ew: pd.DataFrame,
    fci_pca: pd.DataFrame,
) -> pd.DataFrame:
    """
    Correlation between EW and PCA indices by country on overlapping sample.
    """
    merged = fci_ew.merge(
        fci_pca,
        on=["date", "country_code"],
        how="inner",
    )

    rows = []
    for country, g in merged.groupby("country_code"):
        corr = g["fci_ew"].corr(g["fci_pca"])
        rows.append(
            {
                "country_code": country,
                "n_obs_overlap": len(g),
                "corr_ew_pca": corr,
                "mean_ew": g["fci_ew"].mean(),
                "mean_pca": g["fci_pca"].mean(),
                "std_ew": g["fci_ew"].std(ddof=0),
                "std_pca": g["fci_pca"].std(ddof=0),
            }
        )

    return pd.DataFrame(rows).sort_values("country_code").reset_index(drop=True)


def summarize_top_episode_by_country(
    episode_df: pd.DataFrame,
    value_col: str = "mean_value",
    prefix: str = "ew",
) -> pd.DataFrame:
    """
    For each country, select the episode with the highest average index level.
    """
    rows = []
    for country, g in episode_df.groupby("country_code"):
        g = g.sort_values(value_col, ascending=False).reset_index(drop=True)
        top = g.iloc[0]

        rows.append(
            {
                "country_code": country,
                f"{prefix}_top_episode": top["episode"],
                f"{prefix}_top_episode_mean": top[value_col],
                f"{prefix}_top_episode_max": top.get("max_value", None),
                f"{prefix}_top_episode_n_obs": top.get("n_obs", None),
            }
        )

    return pd.DataFrame(rows).sort_values("country_code").reset_index(drop=True)


def summarize_top_peak_by_country(
    peaks_df: pd.DataFrame,
    value_col: str = "peak_value",
    prefix: str = "ew",
) -> pd.DataFrame:
    """
    For each country, select the single largest peak.
    """
    rows = []
    for country, g in peaks_df.groupby("country_code"):
        g = g.sort_values(value_col, ascending=False).reset_index(drop=True)
        top = g.iloc[0]

        rows.append(
            {
                "country_code": country,
                f"{prefix}_top_peak_date": top["date"],
                f"{prefix}_top_peak_value": top[value_col],
                f"{prefix}_top_peak_episode": top.get("episode", ""),
                f"{prefix}_top_peak_label": top.get("window_label", ""),
            }
        )

    return pd.DataFrame(rows).sort_values("country_code").reset_index(drop=True)


def summarize_pca_loading_importance(
    pca_loadings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize PCA loading importance across countries.
    """
    req = {"country_code", "indicator_code", "loading"}
    missing = req - set(pca_loadings.columns)
    if missing:
        raise ValueError(f"Missing columns in pca_loadings: {missing}")

    out = (
        pca_loadings.groupby("indicator_code", as_index=False)["loading"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "avg_loading", "median": "median_loading", "std": "std_loading"})
    )

    out["abs_avg_loading"] = out["avg_loading"].abs()
    out = out.sort_values("abs_avg_loading", ascending=False).reset_index(drop=True)
    return out


def summarize_index_coverage(
    fci_ew: pd.DataFrame,
    fci_pca: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize historical coverage of EW and PCA by country.
    """
    rows = []
    countries = sorted(set(fci_ew["country_code"]).union(set(fci_pca["country_code"])))

    for country in countries:
        g_ew = fci_ew[fci_ew["country_code"] == country].copy()
        g_pca = fci_pca[fci_pca["country_code"] == country].copy()

        rows.append(
            {
                "country_code": country,
                "ew_start": g_ew["date"].min() if not g_ew.empty else pd.NaT,
                "ew_end": g_ew["date"].max() if not g_ew.empty else pd.NaT,
                "ew_n_obs": len(g_ew),
                "pca_start": g_pca["date"].min() if not g_pca.empty else pd.NaT,
                "pca_end": g_pca["date"].max() if not g_pca.empty else pd.NaT,
                "pca_n_obs": len(g_pca),
            }
        )

    return pd.DataFrame(rows).sort_values("country_code").reset_index(drop=True)