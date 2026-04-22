from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.validation.windows import (
    get_baseline_stress_windows,
    label_windows,
)
from src.validation.metrics import (
    compare_stress_vs_tranquil,
    extract_top_peaks,
    summarize_episode_means,
)
from src.validation.plots import (
    plot_index_with_windows,
    plot_peak_markers,
)


def ensure_dirs() -> None:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures_wp4").mkdir(parents=True, exist_ok=True)


def load_index(path: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    req = {"date", "country_code", value_col}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df.sort_values(["country_code", "date"]).reset_index(drop=True)


def main() -> None:
    ensure_dirs()

    # 1. Inputs from WP3
    fci_ew = load_index("data/processed/fci_ew.csv", "fci_ew")
    fci_pca = load_index("data/processed/fci_pca.csv", "fci_pca")

    # 2. Stress windows (baseline narrative windows)
    stress_windows = get_baseline_stress_windows()

    # 3. Label windows
    ew_labeled = label_windows(
        fci_ew,
        date_col="date",
        country_col="country_code",
        value_col="fci_ew",
        stress_windows=stress_windows,
        transition_buffer_months=3,
    )

    pca_labeled = label_windows(
        fci_pca,
        date_col="date",
        country_col="country_code",
        value_col="fci_pca",
        stress_windows=stress_windows,
        transition_buffer_months=3,
    )

    ew_labeled.to_csv("data/processed/wp4_fci_ew_labeled.csv", index=False)
    pca_labeled.to_csv("data/processed/wp4_fci_pca_labeled.csv", index=False)

    # 4. Stress vs tranquil comparisons
    ew_compare = compare_stress_vs_tranquil(
        ew_labeled,
        value_col="fci_ew",
        label_col="window_label",
    )
    pca_compare = compare_stress_vs_tranquil(
        pca_labeled,
        value_col="fci_pca",
        label_col="window_label",
    )

    ew_compare.to_csv("data/processed/wp4_compare_ew.csv", index=False)
    pca_compare.to_csv("data/processed/wp4_compare_pca.csv", index=False)

    # 5. Episode means by country
    ew_episode_means = summarize_episode_means(
        ew_labeled,
        value_col="fci_ew",
        stress_windows=stress_windows,
    )
    pca_episode_means = summarize_episode_means(
        pca_labeled,
        value_col="fci_pca",
        stress_windows=stress_windows,
    )

    ew_episode_means.to_csv("data/processed/wp4_episode_means_ew.csv", index=False)
    pca_episode_means.to_csv("data/processed/wp4_episode_means_pca.csv", index=False)

    # 6. Peak extraction
    ew_peaks = extract_top_peaks(
        ew_labeled,
        value_col="fci_ew",
        top_n=10,
        min_spacing_months=6,
    )
    pca_peaks = extract_top_peaks(
        pca_labeled,
        value_col="fci_pca",
        top_n=10,
        min_spacing_months=6,
    )

    ew_peaks.to_csv("data/processed/wp4_peaks_ew.csv", index=False)
    pca_peaks.to_csv("data/processed/wp4_peaks_pca.csv", index=False)

    # 7. Figures
    countries = sorted(set(fci_ew["country_code"]).union(set(fci_pca["country_code"])))

    for c in countries:
        plot_index_with_windows(
            ew_labeled,
            country=c,
            value_col="fci_ew",
            stress_windows=stress_windows,
            title=f"EW index with stress windows — {c}",
            output_path=f"outputs/figures_wp4/fci_ew_windows_{c}.pdf",
        )

        plot_index_with_windows(
            pca_labeled,
            country=c,
            value_col="fci_pca",
            stress_windows=stress_windows,
            title=f"PCA index with stress windows — {c}",
            output_path=f"outputs/figures_wp4/fci_pca_windows_{c}.pdf",
        )

        plot_peak_markers(
            ew_labeled,
            ew_peaks,
            country=c,
            value_col="fci_ew",
            title=f"Top EW peaks — {c}",
            output_path=f"outputs/figures_wp4/fci_ew_peaks_{c}.pdf",
        )

        plot_peak_markers(
            pca_labeled,
            pca_peaks,
            country=c,
            value_col="fci_pca",
            title=f"Top PCA peaks — {c}",
            output_path=f"outputs/figures_wp4/fci_pca_peaks_{c}.pdf",
        )

    print("WP4 run finished.")


if __name__ == "__main__":
    main()