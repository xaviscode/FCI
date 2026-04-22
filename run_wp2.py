from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from tqdm import tqdm

from src.data_acquisition.bis_portal import BisPortalClient
from src.data_acquisition.fred import FredClient
from src.processing.db import ensure_db, insert_observations, upsert_series_metadata
from src.processing.harmonize import (
    aggregate_to_monthly,
    compute_missingness_by_freq,
    normalize_quarterly_dates,
    quarterly_to_monthly_step,
    term_spread,
    yoy_pct_monthly,
    yoy_pct_quarterly,
)
from src.processing.metadata_exports import (
    write_acquisition_log,
    write_data_dictionary,
    write_feasibility_matrix,
)


def load_config(path: str) -> dict:
    """
    Load YAML config into a Python dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cfg_hash(cfg: dict) -> str:
    """
    Short stable hash of config for run reproducibility logging.
    """
    b = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:12]


def iso_now() -> str:
    """
    Current timestamp in ISO format.
    """
    return datetime.now().isoformat(timespec="seconds")


def make_series_id(ind_code: str, country: str, source: str, source_code: str) -> str:
    """
    Unique deterministic series id.
    """
    return f"{ind_code}__{country}__{source}__{source_code}".replace(" ", "_")


def ensure_dirs(cfg: dict) -> None:
    """
    Create the expected directory structure.
    """
    base = Path(cfg["project"]["base_dir"])
    for k in ["raw_dir", "interim_dir", "processed_dir", "metadata_dir"]:
        (base / cfg["storage"][k]).mkdir(parents=True, exist_ok=True)
    (base / "outputs/figures_wp2").mkdir(parents=True, exist_ok=True)


# Main
def main() -> None:
    cfg = load_config("src/config/wp2_config.yaml")
    ensure_dirs(cfg)

    db_path = cfg["storage"]["sqlite_path"]
    ensure_db(db_path)

    gate = cfg["wp1_gate_rules"]
    min_years = gate["min_years_history"]
    max_miss = gate["max_missingness"]

    acquisition_rows: List[dict] = []
    dictionary_rows: List[dict] = []
    feasibility_rows: List[dict] = []
    processed_long_rows: List[dict] = []

    fred = FredClient(api_key="9d1f78ce7f15e525bd0b74a5258dca83")
    bis = BisPortalClient()

    countries = [c["code"] for c in cfg["countries"]]
    indicators = cfg["indicators"]

    for ind in tqdm(indicators, desc="Indicators"):
        ind_code = ind["code"]
        ind_name = ind["name"]
        direction = int(ind["direction"])
        block = ind.get("block", "")

        for country in countries:
            used_series_code = ""
            is_fallback = 0
            reason = ""
            status = "ok"

            try:
                preferred = ind["preferred"]
                params: Dict[str, Any] = preferred["params"]
                source: str = preferred["source"]
                transform: str = ind["transform"]
                input_freq: str = params.get("input_freq", "M")
                agg: str = params.get("agg", "mean")

                # 1. FETCH
                fetched: Dict[str, pd.DataFrame] = {}

                if source == "FRED":
                    if transform == "term_spread":
                        s10 = params["series_10y_by_country"][country]
                        s3m = params["series_3m_by_country"][country]
                        used_series_code = f"{s10}-{s3m}"
                        fetched["10y"] = fred.fetch_series(s10)
                        fetched["3m"] = fred.fetch_series(s3m)
                    else:
                        sid = params["series_id_by_country"][country] if "series_id_by_country" in params else params["series_id"]
                        used_series_code = sid
                        fetched["main"] = fred.fetch_series(sid)

                        if input_freq == "Q":
                            fetched["main"] = normalize_quarterly_dates(fetched["main"])

                elif source == "BIS_PORTAL":
                    dataset = params["dataset"]
                    key = params["key_by_country"][country]
                    used_series_code = f"{dataset}:{key}"
                    fetched["main"] = bis.fetch_series(dataset=dataset, key=key)

                    if input_freq == "Q":
                        fetched["main"] = normalize_quarterly_dates(fetched["main"])

                else:
                    raise ValueError(f"Unknown source: {source}")

                # 2. ALIGN
                def to_monthly(df_in: pd.DataFrame) -> pd.DataFrame:
                    """
                    Convert raw series to monthly timeline.
                    Quarterly series use step-function mapping (carry-forward).
                    Daily/weekly/monthly series aggregate to monthly using mean/last.
                    """
                    if input_freq == "Q":
                        return quarterly_to_monthly_step(df_in)
                    return aggregate_to_monthly(df_in, how=agg)

                # 3. TRANSFORM
                miss_df: pd.DataFrame
                miss_freq: str

                if transform == "term_spread":
                    m10 = aggregate_to_monthly(fetched["10y"], how=agg)
                    m3m = aggregate_to_monthly(fetched["3m"], how=agg)
                    monthly = term_spread(m10, m3m)
                    miss_df, miss_freq = monthly, "M"

                elif transform == "level":
                    monthly = to_monthly(fetched["main"])
                    miss_df, miss_freq = (fetched["main"], "Q") if input_freq == "Q" else (monthly, "M")

                elif transform == "yoy_pct":
                    if input_freq == "Q":
                        q_yoy = yoy_pct_quarterly(fetched["main"], periods_q=4)
                        monthly = quarterly_to_monthly_step(q_yoy)
                        miss_df, miss_freq = q_yoy, "Q"
                    else:
                        m_level = to_monthly(fetched["main"])
                        monthly = yoy_pct_monthly(m_level, periods_m=12)
                        miss_df, miss_freq = monthly, "M"
                else:
                    raise ValueError(f"Unknown transform: {transform}")

                if monthly.empty:
                    raise ValueError("Empty series after transform (no observations).")

                # 4. QA metrics
                start = pd.to_datetime(monthly["date"]).min()
                end = pd.to_datetime(monthly["date"]).max()
                years = (end - start).days / 365.25

                miss_start = pd.to_datetime(miss_df["date"]).min()
                miss_end = pd.to_datetime(miss_df["date"]).max()
                miss = compute_missingness_by_freq(miss_df, miss_freq, miss_start, miss_end)

                if years < min_years:
                    status = "rejected"
                    reason = f"History {years:.1f}y < {min_years}y"

                if miss > max_miss:
                    status = "rejected"
                    reason = (reason + "; " if reason else "") + f"Missingness {miss:.2%} > {max_miss:.0%}"

                # 5. Store to SQLite
                sid_full = make_series_id(ind_code, country, source, used_series_code)

                meta = {
                    "series_id": sid_full,
                    "indicator_code": ind_code,
                    "country_code": country,
                    "source_name": source,
                    "source_series_code": used_series_code,
                    "frequency": "M",
                    "units": "",
                    "direction": direction,
                    "transform_spec": transform,
                    "start_date": str(start.date()),
                    "end_date": str(end.date()),
                    "missingness": float(miss),
                    "is_fallback": int(is_fallback),
                    "notes": "" if status == "ok" else f"{status}: {reason}",
                }
                upsert_series_metadata(db_path, meta)

                if status == "ok":
                    obs = monthly.copy()
                    obs["date"] = pd.to_datetime(obs["date"]).dt.strftime("%Y-%m-%d")
                    insert_observations(db_path, sid_full, obs)

                    for _, r in obs.iterrows():
                        processed_long_rows.append(
                            {
                                "date": r["date"],
                                "country_code": country,
                                "indicator_code": ind_code,
                                "value": r["value"],
                            }
                        )

                # 6. Logs and metadata
                acquisition_rows.append(
                    {
                        "timestamp": iso_now(),
                        "indicator_code": ind_code,
                        "country_code": country,
                        "source": source,
                        "series_code": used_series_code,
                        "status": status,
                        "reason": reason,
                        "is_fallback": is_fallback,
                    }
                )

                dictionary_rows.append(
                    {
                        "code": ind_code,
                        "name": ind_name,
                        "block": block,
                        "definition": "",
                        "source": source,
                        "freq": "M",
                        "units": "",
                        "direction": direction,
                        "transform": transform,
                        "availability": "",
                        "coverage": f"{meta['start_date']}..{meta['end_date']}",
                        "issues": meta["notes"],
                    }
                )

                feasibility_rows.append(
                    {
                        "indicator_code": ind_code,
                        "country_code": country,
                        "status": status,
                        "used_source": source,
                        "used_series_code": used_series_code,
                        "is_fallback": is_fallback,
                        "reason": reason,
                    }
                )

            except Exception as e:
                acquisition_rows.append(
                    {
                        "timestamp": iso_now(),
                        "indicator_code": ind_code,
                        "country_code": country,
                        "source": ind["preferred"]["source"],
                        "series_code": used_series_code,
                        "status": "error",
                        "reason": str(e),
                        "is_fallback": 0,
                    }
                )
                feasibility_rows.append(
                    {
                        "indicator_code": ind_code,
                        "country_code": country,
                        "status": "error",
                        "used_source": ind["preferred"]["source"],
                        "used_series_code": used_series_code,
                        "is_fallback": 0,
                        "reason": str(e),
                    }
                )

    # 7. Export processed datasets
    out_long = Path(cfg["outputs"]["processed_long"])
    out_wide = Path(cfg["outputs"]["processed_wide"])
    out_long.parent.mkdir(parents=True, exist_ok=True)

    panel_long = pd.DataFrame(processed_long_rows).sort_values(["date", "country_code", "indicator_code"])
    panel_long.to_csv(out_long, index=False)

    if not panel_long.empty:
        panel_wide = (
            panel_long.pivot_table(index=["date", "country_code"], columns="indicator_code", values="value")
            .reset_index()
            .sort_values(["date", "country_code"])
        )
        panel_wide.to_csv(out_wide, index=False)

    # 8. Export metadata files
    write_acquisition_log(cfg["outputs"]["acquisition_log"], acquisition_rows)
    write_data_dictionary(cfg["outputs"]["data_dictionary"], _dedupe_by_code(dictionary_rows))
    write_feasibility_matrix(cfg["outputs"]["feasibility_matrix"], feasibility_rows)

    print("WP2 run finished.")
    print(f"Processed long: {out_long}")
    print(f"Processed wide: {out_wide}")
    print(f"SQLite DB: {db_path}")
    print(f"Config hash: {cfg_hash(cfg)}")


def _dedupe_by_code(rows: List[dict]) -> List[dict]:
    """
    Keep the last row per indicator code (simple dedupe for dictionary export).
    """
    by: Dict[str, dict] = {}
    for r in rows:
        by[r["code"]] = r
    return list(by.values())


if __name__ == "__main__":
    main()