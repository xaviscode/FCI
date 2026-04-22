from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml

from src.index.standardize import (
    winsorize_by_group,
    zscore_by_group,
    apply_direction_alignment,
)
from src.index.construct import (
    build_block_subindices,
    build_equal_weight_index,
    build_pca_index,
)
from src.index.plots import (
    plot_country_index,
    plot_country_blocks,
    plot_pca_loadings,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures_wp3").mkdir(parents=True, exist_ok=True)


def build_indicator_metadata(cfg: dict) -> pd.DataFrame:
    """
    Extract indicator metadata from wp2_config.yaml so WP3 knows:
    - block
    - direction
    """
    rows = []
    for ind in cfg["indicators"]:
        rows.append(
            {
                "indicator_code": ind["code"],
                "block": ind["block"],
                "direction": int(ind["direction"]),
                "transform": ind["transform"],
                "name": ind["name"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()

    cfg = load_config("src/config/wp2_config.yaml")
    meta = build_indicator_metadata(cfg)

    panel_long_path = Path("data/processed/panel_long.csv")
    if not panel_long_path.exists():
        raise FileNotFoundError("Missing data/processed/panel_long.csv. Run WP2 first.")

    panel = pd.read_csv(panel_long_path)
    panel["date"] = pd.to_datetime(panel["date"])

    required_cols = {"date", "country_code", "indicator_code", "value"}
    missing_cols = required_cols - set(panel.columns)
    if missing_cols:
        raise ValueError(f"panel_long.csv is missing columns: {missing_cols}")

    df = panel.merge(meta, on="indicator_code", how="left")
    if df["direction"].isna().any():
        missing_inds = df.loc[df["direction"].isna(), "indicator_code"].drop_duplicates().tolist()
        raise ValueError(f"Missing metadata for indicators: {missing_inds}")

    # 1. Direction alignment
    df = apply_direction_alignment(df)

    # 2. Winsorization
    df = winsorize_by_group(
        df,
        value_col="value_aligned",
        group_cols=["country_code", "indicator_code"],
        lower_q=0.01,
        upper_q=0.99,
        out_col="value_winsor",
    )

    # 3. Standardization
    df = zscore_by_group(
        df,
        value_col="value_winsor",
        group_cols=["country_code", "indicator_code"],
        out_col="z_value",
    )

    wp3_panel = df[
        [
            "date",
            "country_code",
            "indicator_code",
            "block",
            "direction",
            "value",
            "value_aligned",
            "value_winsor",
            "z_value",
        ]
    ].copy()
    wp3_panel.to_csv("data/processed/wp3_panel_z.csv", index=False)

    # 4. Block subindices
    block_sub = build_block_subindices(df)
    block_sub.to_csv("data/processed/block_subindices.csv", index=False)

    # 5. Equal-weight FCI
    fci_ew, contributions_ew = build_equal_weight_index(
        df,
        use_block_average=True,
    )
    fci_ew.to_csv("data/processed/fci_ew.csv", index=False)
    contributions_ew.to_csv("data/processed/contributions_ew.csv", index=False)

    # 6. PCA FCI
    fci_pca, pca_loadings = build_pca_index(df)
    fci_pca.to_csv("data/processed/fci_pca.csv", index=False)
    pca_loadings.to_csv("data/processed/pca_loadings.csv", index=False)

    # 7. Plots
    countries = sorted(df["country_code"].dropna().unique().tolist())

    for c in countries:
        plot_country_index(
            fci_ew,
            country=c,
            value_col="fci_ew",
            title=f"FCI Equal-Weight — {c}",
            output_path=f"outputs/figures_wp3/fci_ew_{c}.pdf",
        )

        plot_country_index(
            fci_pca,
            country=c,
            value_col="fci_pca",
            title=f"FCI PCA — {c}",
            output_path=f"outputs/figures_wp3/fci_pca_{c}.pdf",
        )

        plot_country_blocks(
            block_sub,
            country=c,
            output_path=f"outputs/figures_wp3/blocks_{c}.pdf",
        )

    plot_pca_loadings(
        pca_loadings,
        output_path="outputs/figures_wp3/pca_loadings.pdf",
    )

    print("WP3 run finished.")


if __name__ == "__main__":
    main()