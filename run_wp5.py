from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.discussion.compare import (
    compare_ew_vs_pca_correlations,
    summarize_top_episode_by_country,
    summarize_top_peak_by_country,
    summarize_pca_loading_importance,
    summarize_index_coverage,
)
from src.discussion.plots import (
    plot_ew_vs_pca_overlay,
    plot_country_episode_bars,
    plot_pca_loading_heatmap,
)


def ensure_dirs() -> None:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures_wp5").mkdir(parents=True, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    ensure_dirs()

    # 1. Inputs from WP3 and WP4
    fci_ew = load_csv("data/processed/fci_ew.csv")
    fci_pca = load_csv("data/processed/fci_pca.csv")
    pca_loadings = load_csv("data/processed/pca_loadings.csv")

    wp4_episode_means_ew = load_csv("data/processed/wp4_episode_means_ew.csv")
    wp4_episode_means_pca = load_csv("data/processed/wp4_episode_means_pca.csv")
    wp4_peaks_ew = load_csv("data/processed/wp4_peaks_ew.csv")
    wp4_peaks_pca = load_csv("data/processed/wp4_peaks_pca.csv")

    # 2. EW vs PCA correlation by country
    corr_table = compare_ew_vs_pca_correlations(fci_ew, fci_pca)
    corr_table.to_csv("data/processed/wp5_compare_ew_pca_corr.csv", index=False)

    # 3. Top episode by country
    top_episode_ew = summarize_top_episode_by_country(
        wp4_episode_means_ew,
        value_col="mean_value",
        prefix="ew",
    )
    top_episode_pca = summarize_top_episode_by_country(
        wp4_episode_means_pca,
        value_col="mean_value",
        prefix="pca",
    )

    top_episode = top_episode_ew.merge(top_episode_pca, on="country_code", how="outer")
    top_episode.to_csv("data/processed/wp5_top_episode_by_country.csv", index=False)

    # 4. Top peak by country
    top_peak_ew = summarize_top_peak_by_country(
        wp4_peaks_ew,
        value_col="peak_value",
        prefix="ew",
    )
    top_peak_pca = summarize_top_peak_by_country(
        wp4_peaks_pca,
        value_col="peak_value",
        prefix="pca",
    )

    top_peak = top_peak_ew.merge(top_peak_pca, on="country_code", how="outer")
    top_peak.to_csv("data/processed/wp5_top_peak_by_country.csv", index=False)

    # 5. PCA loading importance
    pca_loading_summary = summarize_pca_loading_importance(pca_loadings)
    pca_loading_summary.to_csv("data/processed/wp5_pca_loading_summary.csv", index=False)

    # 6. Index coverage
    coverage = summarize_index_coverage(fci_ew, fci_pca)
    coverage.to_csv("data/processed/wp5_index_coverage.csv", index=False)

    # 7. Figures
    countries = sorted(set(fci_ew["country_code"]).union(set(fci_pca["country_code"])))

    for c in countries:
        plot_ew_vs_pca_overlay(
            fci_ew,
            fci_pca,
            country=c,
            output_path=f"outputs/figures_wp5/ew_vs_pca_{c}.pdf",
        )

    plot_country_episode_bars(
        wp4_episode_means_ew,
        value_col="mean_value",
        title="Average EW index by stress episode",
        output_path="outputs/figures_wp5/episode_means_ew.pdf",
    )

    plot_country_episode_bars(
        wp4_episode_means_pca,
        value_col="mean_value",
        title="Average PCA index by stress episode",
        output_path="outputs/figures_wp5/episode_means_pca.pdf",
    )

    plot_pca_loading_heatmap(
        pca_loadings,
        output_path="outputs/figures_wp5/pca_loading_heatmap.pdf",
    )

    print("WP5 run finished.")


if __name__ == "__main__":
    main()