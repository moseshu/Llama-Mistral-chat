from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def _try_parse_dates(df: pd.DataFrame, candidate_columns: Sequence[str]) -> pd.DataFrame:
    """Attempt to parse datetime columns in-place for given candidate column names.

    Parsing is best-effort and tolerant to errors (coerce to NaT when failing).
    """
    for column in candidate_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def load_dataset(input_path: str, date_column: Optional[str] = None) -> pd.DataFrame:
    """Load a dataset from CSV or JSON into a pandas DataFrame.

    - CSV: uses pandas.read_csv with UTF-8 by default and automatic dtype inference
    - JSON: accepts array-of-objects JSON via pandas.read_json

    Args:
        input_path: Path to CSV or JSON file
        date_column: Optional date column name to parse into datetime

    Returns:
        DataFrame containing the loaded data
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep, engine="python")
    elif suffix == ".json":
        # Support array-of-objects JSON
        df = pd.read_json(path, orient="records", lines=False)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

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

