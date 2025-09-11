from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from .extractors import load_any_table


def _try_parse_dates(df: pd.DataFrame, candidate_columns: Sequence[str]) -> pd.DataFrame:
    """Attempt to parse datetime columns in-place for given candidate column names.

    Parsing is best-effort and tolerant to errors (coerce to NaT when failing).
    """
    for column in candidate_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def load_dataset(input_path: str, date_column: Optional[str] = None) -> pd.DataFrame:
    """Load a dataset from many common formats into a pandas DataFrame.

    Supports CSV/TSV, JSON (array), Excel, HTML, DOCX, PDF (best-effort table extraction).
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = load_any_table(str(path))

    # Best-effort date parsing
    candidates = [date_column] if date_column else []
    # Heuristic fallbacks
    candidates += [
        c
        for c in ["date", "datetime", "timestamp", "time"]
        if c in df.columns and c not in candidates
    ]
    df = _try_parse_dates(df, candidates)

    return df

