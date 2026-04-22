from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_index_with_windows(
    df: pd.DataFrame,
    country: str,
    value_col: str,
    stress_windows: list[dict],
    title: str,
    output_path: str,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    g = df[df["country_code"] == country].copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date")

    plt.figure(figsize=(10, 4.5))
    plt.plot(g["date"], g[value_col], linewidth=1.5)

    for w in stress_windows:
        start = pd.to_datetime(w["start"])
        end = pd.to_datetime(w["end"])
        plt.axvspan(start, end, alpha=0.15)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_peak_markers(
    df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    country: str,
    value_col: str,
    title: str,
    output_path: str,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    g = df[df["country_code"] == country].copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date")

    p = peaks_df[peaks_df["country_code"] == country].copy()
    if not p.empty:
        p["date"] = pd.to_datetime(p["date"])

    plt.figure(figsize=(10, 4.5))
    plt.plot(g["date"], g[value_col], linewidth=1.5)

    if not p.empty:
        plt.scatter(p["date"], p["peak_value"], s=25)
        for _, r in p.iterrows():
            plt.annotate(
                r["date"].strftime("%Y-%m"),
                (r["date"], r["peak_value"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()