from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from .ingest import load_dataset
from .analyze import analyze
from .visualize import (
    plot_time_series,
    plot_top_categories,
    plot_numeric_histograms,
    plot_adaptive,
)
from .report import render_html
from .autodetect import detect_date_column, detect_metric_column


def run_pipeline(
    input_path: str,
    output_dir: str,
    date_column: Optional[str] = None,
    metric: Optional[str] = None,
    title: str = "Data Report",
) -> str:
    df = load_dataset(input_path, date_column=date_column)
    inferred_date = date_column or detect_date_column(df)
    inferred_metric = metric or detect_metric_column(df)
    analysis = analyze(df, date_column=inferred_date, metric_preference=inferred_metric)

    charts = {}
    ts_chart = plot_time_series(analysis.time_series, output_dir)
    if ts_chart:
        charts["timeseries"] = ts_chart
    cat_charts = plot_top_categories(analysis.category_top_counts, output_dir)
    # namespaced as top_{col}
    charts.update({f"top_{k}": v for k, v in cat_charts.items()})
    # histograms not embedded directly but useful
    plot_numeric_histograms(df, output_dir)
    charts.update(plot_adaptive(df, output_dir))

    html_path = render_html(analysis, charts, output_dir, title=title)
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a data analysis HTML report from CSV/JSON")
    parser.add_argument("input", help="Path to input CSV or JSON file")
    parser.add_argument("--out", dest="out", default="./report_output", help="Output directory for artifacts")
    parser.add_argument("--date", dest="date", default=None, help="Date column name for time series resampling (auto-detected if omitted)")
    parser.add_argument("--metric", dest="metric", default=None, help="Preferred metric column to aggregate (auto-detected if omitted)")
    parser.add_argument("--title", dest="title", default="Data Report", help="Report title")

    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = run_pipeline(args.input, str(out_dir), date_column=args.date, metric=args.metric, title=args.title)
    print(html_path)


if __name__ == "__main__":
    main()

