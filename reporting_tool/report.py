from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .analyze import AnalysisResult


def render_html(
    analysis: AnalysisResult,
    charts: Dict[str, str],
    output_dir: str,
    title: str = "Data Report",
) -> str:
    """Render an HTML report into output_dir and return the HTML file path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html")

    # Convert numeric_summary to a 2D table-friendly structure
    numeric_summary = analysis.numeric_summary
    time_series_rows = []
    if analysis.time_series is not None and not analysis.time_series.empty:
        time_series_rows = [
            {"date": str(idx.date()), "value": float(val)}
            for idx, val in analysis.time_series["value"].items()
        ]

    content = template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        title=title,
        numeric_summary=numeric_summary,
        category_top_counts=analysis.category_top_counts,
        category_top_metric=analysis.category_top_metric,
        anomalies=[
            {"date": str(d.date()), "value": v, "z": round(z, 2)}
            for (d, v, z) in analysis.anomalies
        ],
        time_series=time_series_rows,
        charts=charts,
        derived_columns=analysis.derived_columns,
    )

    out_file = out_dir / "report.html"
    out_file.write_text(content, encoding="utf-8")
    return str(out_file)

