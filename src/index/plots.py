from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_country_index(
    df: pd.DataFrame,
    country: str,
    value_col: str,
    title: str,
    output_path: str,
) -> None:
    """
    Plot one index series for one country.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    g = df[df["country_code"] == country].copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date")

    plt.figure(figsize=(10, 4.5))
    plt.plot(g["date"], g[value_col])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_country_blocks(
    block_df: pd.DataFrame,
    country: str,
    output_path: str,
) -> None:
    """
    Plot all block subindices for one country.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    g = block_df[block_df["country_code"] == country].copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date")

    plt.figure(figsize=(10, 5))
    for block, sub in g.groupby("block"):
        plt.plot(sub["date"], sub["block_value"], label=block)

    plt.title(f"Block subindices — {country}")
    plt.xlabel("Date")
    plt.ylabel("Block index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pca_loadings(
    loadings_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot average PCA loadings across countries by indicator.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if loadings_df.empty:
        return

    avg = (
        loadings_df.groupby("indicator_code", as_index=False)["loading"]
        .mean()
        .sort_values("loading")
    )

    plt.figure(figsize=(10, 6))
    plt.barh(avg["indicator_code"], avg["loading"])
    plt.title("Average PCA loadings across countries")
    plt.xlabel("Loading")
    plt.ylabel("Indicator")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()