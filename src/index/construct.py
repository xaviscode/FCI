from __future__ import annotations

import numpy as np
import pandas as pd


def build_block_subindices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average z-scored indicators within each block, by country and date.
    """
    req = {"date", "country_code", "block", "indicator_code", "z_value"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for block subindices: {missing}")

    out = (
        df.groupby(["date", "country_code", "block"], as_index=False)["z_value"]
        .mean()
        .rename(columns={"z_value": "block_value"})
    )
    return out


def build_equal_weight_index(
    df: pd.DataFrame,
    use_block_average: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the equal-weight index.

    If use_block_average=True:
      1) average within block
      2) average across blocks
    This prevents blocks with more indicators from dominating.

    Returns:
      fci_ew: date x country x fci_ew
      contributions_ew: date x country x block x block_value
    """
    req = {"date", "country_code", "block", "indicator_code", "z_value"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for EW index: {missing}")

    if use_block_average:
        block_sub = build_block_subindices(df)
        fci = (
            block_sub.groupby(["date", "country_code"], as_index=False)["block_value"]
            .mean()
            .rename(columns={"block_value": "fci_ew"})
        )
        contributions = block_sub.copy()
        return fci, contributions

    fci = (
        df.groupby(["date", "country_code"], as_index=False)["z_value"]
        .mean()
        .rename(columns={"z_value": "fci_ew"})
    )

    contributions = build_block_subindices(df)
    return fci, contributions


def _first_pc_from_wide(X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Compute first principal component from a wide matrix with rows=time, cols=indicators.

    Returns:
      scores: time-indexed PC1 scores
      loadings: indicator-indexed PC1 loadings
    """
    X = X.dropna(axis=0, how="any").copy()
    if X.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc.values, full_matrices=False)

    scores = pd.Series(U[:, 0] * S[0], index=X.index, name="pc1")
    loadings = pd.Series(Vt[0, :], index=X.columns, name="loading")

    if loadings.mean() < 0:
        scores = -scores
        loadings = -loadings

    return scores, loadings


def build_pca_index(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build PCA-based index per country.

    Returns:
      fci_pca: date x country x fci_pca
      pca_loadings: country x indicator x loading
    """
    req = {"date", "country_code", "indicator_code", "z_value"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for PCA index: {missing}")

    fci_rows = []
    load_rows = []

    for country, g in df.groupby("country_code"):
        wide = (
            g.pivot_table(
                index="date",
                columns="indicator_code",
                values="z_value",
                aggfunc="mean",
            )
            .sort_index()
        )

        scores, loadings = _first_pc_from_wide(wide)

        if not scores.empty:
            for dt, val in scores.items():
                fci_rows.append(
                    {
                        "date": dt,
                        "country_code": country,
                        "fci_pca": val,
                    }
                )

        if not loadings.empty:
            for ind_code, val in loadings.items():
                load_rows.append(
                    {
                        "country_code": country,
                        "indicator_code": ind_code,
                        "loading": val,
                    }
                )

    fci_pca = pd.DataFrame(fci_rows).sort_values(["date", "country_code"])
    pca_loadings = pd.DataFrame(load_rows).sort_values(["country_code", "indicator_code"])

    return fci_pca, pca_loadings