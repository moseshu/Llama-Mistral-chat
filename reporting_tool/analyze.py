from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class AnalysisResult:
    numeric_summary: Dict[str, Dict[str, float]]
    category_top_counts: Dict[str, List[Tuple[str, int]]]
    category_top_metric: Dict[str, List[Tuple[str, float]]]
    time_series: Optional[pd.DataFrame]
    anomalies: List[Tuple[pd.Timestamp, float, float]]  # (date, value, zscore)
    derived_columns: List[str]


def _compute_numeric_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    numeric_df = df.select_dtypes(include=["number"])  # type: ignore[arg-type]
    if numeric_df.empty:
        return {}
    summary: Dict[str, Dict[str, float]] = {}
    for col in numeric_df.columns:
        s = numeric_df[col].dropna()
        if s.empty:
            continue
        summary[col] = {
            "count": float(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
            "min": float(s.min()),
            "median": float(s.median()),
            "max": float(s.max()),
        }
    return summary


def _ensure_revenue(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    derived: List[str] = []
    if all(c in df.columns for c in ["units", "price"]) and "revenue" not in df.columns:
        df = df.copy()
        df["revenue"] = df["units"].astype(float) * df["price"].astype(float)
        derived.append("revenue")
    return df, derived


def _compute_category_tops(df: pd.DataFrame, metric_column: Optional[str], top_n: int) -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[str, List[Tuple[str, float]]]]:
    cat_df = df.select_dtypes(include=["object", "category"])  # type: ignore[arg-type]
    if cat_df.empty:
        return {}, {}
    top_counts: Dict[str, List[Tuple[str, int]]] = {}
    top_metric: Dict[str, List[Tuple[str, float]]] = {}
    for col in cat_df.columns:
        counts = (
            df[col]
            .astype("string")
            .fillna("<NA>")
            .value_counts(dropna=False)
            .head(top_n)
        )
        top_counts[col] = [(str(idx), int(val)) for idx, val in counts.items()]

        if metric_column and metric_column in df.columns:
            metric_series = (
                df.groupby(col)[metric_column]
                .sum(numeric_only=True)
                .sort_values(ascending=False)
                .head(top_n)
            )
            top_metric[col] = [(str(idx), float(val)) for idx, val in metric_series.items()]
    return top_counts, top_metric


def _compute_time_series(df: pd.DataFrame, date_column: Optional[str], metric_column: Optional[str]) -> Optional[pd.DataFrame]:
    if not date_column or date_column not in df.columns:
        return None
    if metric_column and metric_column in df.columns:
        value_col = metric_column
    else:
        # Fallback to first numeric column
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        if not numeric_cols:
            return None
        value_col = numeric_cols[0]

    ts = (
        df[[date_column, value_col]]
        .dropna(subset=[date_column])
        .sort_values(date_column)
        .set_index(date_column)
        .resample("D")
        .sum(numeric_only=True)
    )
    ts = ts.rename(columns={value_col: "value"})
    ts.index.name = "date"
    return ts


def _detect_anomalies(ts: Optional[pd.DataFrame], z_threshold: float = 2.5) -> List[Tuple[pd.Timestamp, float, float]]:
    if ts is None or ts.empty:
        return []
    s = ts["value"].astype(float)
    mu = float(s.mean())
    sigma = float(s.std(ddof=1)) if s.count() > 1 else 0.0
    if sigma == 0.0:
        return []
    zscores = (s - mu) / sigma
    mask = zscores.abs() > z_threshold
    out: List[Tuple[pd.Timestamp, float, float]] = []
    for idx, val in s[mask].items():
        out.append((pd.Timestamp(idx), float(val), float(zscores.loc[idx])))
    return out


def analyze(df: pd.DataFrame, date_column: Optional[str] = None, metric_preference: Optional[str] = None, top_n: int = 5) -> AnalysisResult:
    df2, derived = _ensure_revenue(df)
    numeric_summary = _compute_numeric_summary(df2)

    metric_column = metric_preference if metric_preference in df2.columns else (
        "revenue" if "revenue" in df2.columns else None
    )

    top_counts, top_metric = _compute_category_tops(df2, metric_column, top_n)
    ts = _compute_time_series(df2, date_column, metric_column)
    anomalies = _detect_anomalies(ts)

    return AnalysisResult(
        numeric_summary=numeric_summary,
        category_top_counts=top_counts,
        category_top_metric=top_metric,
        time_series=ts,
        anomalies=anomalies,
        derived_columns=derived,
    )

