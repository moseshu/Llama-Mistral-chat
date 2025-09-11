from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd


DATE_CANDIDATES = [
    "date",
    "datetime",
    "timestamp",
    "time",
    "created_at",
    "updated_at",
]

METRIC_KEYWORDS = [
    "revenue",
    "amount",
    "total",
    "value",
    "count",
    "qty",
    "quantity",
    "score",
]


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    # Direct name matches first
    for c in DATE_CANDIDATES:
        if c in df.columns:
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except Exception:
                continue
    # Try any column that can be parsed as datetime with low NaT rate
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            nat_rate = parsed.isna().mean()
            if nat_rate < 0.2:  # at least 80% parsable
                return col
        except Exception:
            continue
    return None


def detect_metric_column(df: pd.DataFrame) -> Optional[str]:
    # Prefer explicitly named columns
    lower_map = {str(c).lower(): c for c in df.columns}
    for kw in METRIC_KEYWORDS:
        if kw in lower_map:
            col = lower_map[kw]
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    # Fallback to first numeric column with variance
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    for col in numeric_cols:
        s = df[col].dropna()
        if s.nunique() > 5 and float(s.std(ddof=1) or 0) > 0:
            return col
    return numeric_cols[0] if numeric_cols else None

