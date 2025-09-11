# Data Reporting Tool

Generate an HTML report from CSV/JSON: load → analyze → chart → render.

## Quickstart

1) Create venv and install deps

```bash
make install
```

2) Run sample report

```bash
make run
```

Output directory: `report_output/`, open `report_output/report.html` in a browser.

## CLI

```bash
python -m reporting_tool.main INPUT --out OUT_DIR --date DATE_COL --metric METRIC --title "Title"
```

- `INPUT`: CSV or JSON file
- `--date`: date column for time series resampling
- `--metric`: preferred metric for aggregation (defaults to `revenue` if derivable)

## Notes
- If both `units` and `price` exist, a `revenue` column is derived.
- Charts saved alongside HTML for portability.
- Minimal anomaly detection via z-score on daily series.

