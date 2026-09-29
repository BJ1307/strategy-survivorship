# Convenience targets. Every target is a plain `python -m strategy_survivorship.<module>`
# command documented in README.md; nothing here is required.
PY ?= .venv/bin/python

.PHONY: test smoke figures reports

## run the test suite (about 30 s)
test:
	$(PY) -m pytest

## small synthetic end-to-end run, no data needed (writes to outputs/smoke)
smoke:
	$(PY) -m strategy_survivorship.stage1 --smoke --out outputs/smoke

## redraw figures from CSVs already under outputs/ -- no simulation
figures:
	$(PY) -m strategy_survivorship.figures_brief
	$(PY) -m strategy_survivorship.stage2d  --figures-only
	$(PY) -m strategy_survivorship.stage2e  --figures-only
	$(PY) -m strategy_survivorship.stage3a  --figures-only
	$(PY) -m strategy_survivorship.stage3a1 --figures-only
	$(PY) -m strategy_survivorship.stage3b  --figures-only

## regenerate reports from CSVs already under outputs/ -- no simulation
reports:
	$(PY) -m strategy_survivorship.stage3a1 --report-only
	$(PY) -m strategy_survivorship.stage3b  --report-only
	$(PY) -m strategy_survivorship.mixed_noise report
