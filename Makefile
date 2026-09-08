# Convenience targets. Nothing here is required: every command is a plain
# `python -m strategy_survivorship.<entry>` and is spelled out in docs/HANDOFF_COMPANY_CLAUDE.md.
PY ?= .venv/bin/python

.PHONY: test figures brief report-only clean-scratch

test:
	$(PY) -m pytest -q

## redraw every figure from the CSVs already in outputs/ -- no simulation runs
figures:
	$(PY) -m strategy_survivorship.figures_brief
	$(PY) -m strategy_survivorship.run_stage2e  --figures-only
	$(PY) -m strategy_survivorship.run_stage3a  --figures-only
	$(PY) -m strategy_survivorship.run_stage3a1 --figures-only
	$(PY) -m strategy_survivorship.run_stage3b  --figures-only

## rebuild the four brief figures only
brief:
	$(PY) -m strategy_survivorship.figures_brief

## regenerate stage reports from existing CSVs -- no simulation runs
report-only:
	$(PY) -m strategy_survivorship.run_stage3a1 --report-only
	$(PY) -m strategy_survivorship.run_stage3b  --report-only

clean-scratch:
	rm -rf outputs/smoke*/ outputs/review_bundle*/ outputs/*.zip outputs/status.json

## rebuild a review bundle for one stage, e.g. `make bundle STAGE=3b`
STAGE ?= 3b
bundle:
	@rm -rf outputs/review_bundle_stage$(STAGE)
	@mkdir -p outputs/review_bundle_stage$(STAGE)/{figures,data,code}
	@cp outputs/stage$(STAGE)_report.md theory.md docs/BRIEF.md \
	    docs/HANDOFF_COMPANY_CLAUDE.md outputs/review_bundle_stage$(STAGE)/
	@cp outputs/figures/fig$(STAGE)*.png outputs/review_bundle_stage$(STAGE)/figures/ 2>/dev/null || true
	@cp docs/figures/*.png outputs/review_bundle_stage$(STAGE)/figures/
	@cp outputs/stage$(STAGE)_*.csv outputs/review_bundle_stage$(STAGE)/data/
	@cp -r src tests pyproject.toml requirements*.txt outputs/review_bundle_stage$(STAGE)/code/
	@cd outputs && zip -qr review_bundle_stage$(STAGE).zip review_bundle_stage$(STAGE)
	@echo "outputs/review_bundle_stage$(STAGE).zip"

.PHONY: bundle
