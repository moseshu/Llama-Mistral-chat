VENV?=.venv
PY?=python3

.PHONY: venv install lint test run clean

venv:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate; pip install --upgrade pip

install: venv
	. $(VENV)/bin/activate; pip install -r requirements.txt
	. $(VENV)/bin/activate; pip install pandas matplotlib jinja2 pytest

lint:
	. $(VENV)/bin/activate; python -m pyflakes reporting_tool || true

test:
	. $(VENV)/bin/activate; pytest -q | cat

run:
	. $(VENV)/bin/activate; $(PY) -m reporting_tool.main ./data/sample_sales.csv --out ./report_output --date date --metric revenue --title "Sales Report"

clean:
	rm -rf $(VENV) report_output .pytest_cache

