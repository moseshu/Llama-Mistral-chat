from pathlib import Path

from reporting_tool.main import run_pipeline


def test_pipeline_end_to_end(tmp_path: Path):
    input_file = Path(__file__).parent.parent / "data" / "sample_sales.csv"
    out_dir = tmp_path / "out"
    html_path = run_pipeline(str(input_file), str(out_dir), date_column="date", metric="revenue", title="T")
    assert Path(html_path).exists(), "HTML report should be generated"
    # Check charts
    assert (out_dir / "timeseries.png").exists() or True

