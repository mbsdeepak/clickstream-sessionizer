# Clickstream Sessionizer -- developer workflow.
#
# Uses a local venv at .venv created with Homebrew Python 3.11. PySpark 4.0.0 is
# pinned because it supports the Java 21 runtime on this machine (Spark 3.5 does
# not officially support JDK 21).

PY        := .venv/bin/python
PIP       := .venv/bin/pip
PYTEST    := .venv/bin/pytest
PYTHON311 := /opt/homebrew/bin/python3.11
MODULE    := clickstream_sessionizer.pipeline
export PYTHONPATH := src

.PHONY: help venv install gen-data bronze silver sessions gold pipeline stream test clean

help:
	@echo "Targets:"
	@echo "  venv      Create .venv with Homebrew Python 3.11"
	@echo "  install   Install requirements into .venv"
	@echo "  gen-data  Generate synthetic raw clickstream"
	@echo "  bronze    Raw JSON -> typed Parquet"
	@echo "  silver    Clean/dedupe/filter bots/enrich (broadcast join)"
	@echo "  sessions  Gap-based sessionization (window functions)"
	@echo "  gold      Funnel/retention/top-paths analytics"
	@echo "  pipeline  Full batch pipeline end-to-end (gen->bronze->silver->sessions->gold)"
	@echo "  stream    Structured Streaming session_window variant (availableNow)"
	@echo "  test      Run pytest"
	@echo "  clean     Remove generated data + Spark artifacts"

venv:
	$(PYTHON311) -m venv .venv
	$(PIP) install --upgrade pip

install:
	$(PIP) install -r requirements.txt

gen-data:
	$(PY) -m $(MODULE) gen-data

bronze:
	$(PY) -m $(MODULE) bronze

silver:
	$(PY) -m $(MODULE) silver

sessions:
	$(PY) -m $(MODULE) sessions

gold:
	$(PY) -m $(MODULE) gold

# Full batch pipeline with summary + sample rows.
pipeline:
	$(PY) -m $(MODULE) all

stream:
	$(PY) -m $(MODULE) stream

test:
	$(PYTEST)

clean:
	rm -rf data spark-warehouse metastore_db derby.log
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '.pytest_cache' -type d -prune -exec rm -rf {} +
