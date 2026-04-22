from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_acquisition_log(path: str, rows: List[dict]) -> None:
    """
    Append-only acquisition log. If the file exists, append; else create.
    """
    _ensure_parent(path)
    df = pd.DataFrame(rows)
    if Path(path).exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def write_data_dictionary(path: str, rows: List[dict]) -> None:
    _ensure_parent(path)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_feasibility_matrix(path: str, rows: List[dict]) -> None:
    _ensure_parent(path)
    pd.DataFrame(rows).to_csv(path, index=False)