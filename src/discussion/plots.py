from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_ew_vs_pca_overlay(
    fci_ew: pd.DataFrame,
    fci_pca: pd.DataFrame,
    country: str,
    output_path: str,
) -> None:
    """
    Overlay EW and PCA indices for one country.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    ew = fci_ew[fci_ew["country_code"] == country].copy()
    pca = fci_pca[fci_pca["country_code"] == country].copy()

    ew["date"] = pd.to_datetime(ew["date"])
    pca["date"] = pd.to_datetime(pca["date"])

    plt.figure(figsize=(10, 4.8))
    plt.plot(ew["date"], ew["fci_ew"], label="EW")
    plt.plot(pca["date"], pca["fci_pca"], label="PCA")
    plt.title(f"EW vs PCA — {country}")
    plt.xlabel("Date")
    plt.ylabel("Index value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_country_episode_bars(
    episode_df: pd.DataFrame,
    value_col: str,
    title: str,
    output_path: str,
) -> None:
    """
    Plot average episode values by country and episode.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df = episode_df.copy()
    df["label"] = df["country_code"] + " | " + df["episode"]

    plt.figure(figsize=(12, 6))
    plt.barh(df["label"], df[value_col])
    plt.title(title)
    plt.xlabel(value_col)
    plt.ylabel("Country | Episode")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pca_loading_heatmap(
    pca_loadings: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Heatmap-like matrix of PCA loadings by country and indicator.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    wide = (
        pca_loadings.pivot_table(
            index="indicator_code",
            columns="country_code",
            values="loading",
            aggfunc="mean",
        )
        .sort_index()
    )

    plt.figure(figsize=(7, 6))
    plt.imshow(wide.values, aspect="auto")
    plt.xticks(range(len(wide.columns)), wide.columns)
    plt.yticks(range(len(wide.index)), wide.index)
    plt.title("PCA loadings by country and indicator")
    plt.colorbar(label="Loading")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()