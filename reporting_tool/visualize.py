from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_time_series(ts: Optional[pd.DataFrame], output_dir: str, title: str = "Time Series") -> Optional[str]:
    if ts is None or ts.empty:
        return None
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    out_path = out_dir / "timeseries.png"

    plt.figure(figsize=(9, 4))
    plt.plot(ts.index, ts["value"], label="value", color="#2b8cbe")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


def plot_top_categories(top_counts: Dict[str, List[tuple]], output_dir: str, top_n: int = 5) -> Dict[str, str]:
    if not top_counts:
        return {}
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    outputs: Dict[str, str] = {}
    for column, pairs in top_counts.items():
        labels = [p[0] for p in pairs][:top_n]
        values = [p[1] for p in pairs][:top_n]
        out_path = out_dir / f"top_{column}.png"

        plt.figure(figsize=(8, 4))
        plt.bar(labels, values, color="#7bccc4")
        plt.title(f"Top {len(labels)} by count: {column}")
        plt.ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        outputs[column] = str(out_path)
    return outputs


def plot_numeric_histograms(df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    numeric_df = df.select_dtypes(include=["number"])  # type: ignore[arg-type]
    if numeric_df.empty:
        return {}
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    outputs: Dict[str, str] = {}
    for column in numeric_df.columns:
        s = numeric_df[column].dropna()
        if s.empty:
            continue
        out_path = out_dir / f"hist_{column}.png"
        plt.figure(figsize=(6, 4))
        plt.hist(s, bins=20, color="#fdae6b", edgecolor="#ffffff")
        plt.title(f"Distribution: {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        outputs[column] = str(out_path)
    return outputs

