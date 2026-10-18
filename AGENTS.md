# Repository guidance for coding agents

This repository hosts a Python 3.12+ supply-chain risk and demand platform.
The documented entry points are `python -m src.cli run-all`,
`uvicorn src.api:app --host 0.0.0.0 --port 8000`, and
`streamlit run src/dashboard.py --server.port 8501`. Install with
`python -m pip install -r requirements.lock` followed by
`python -m pip install --no-deps -e .`. The lockfile is compiled from the
pinned manifest with
`uv pip compile pyproject.toml --extra dev --universal --output-file requirements.lock`.

- Use `PLATFORM_DATA_DIR` and `PLATFORM_ARTIFACT_DIR` for runtime output; do not
  commit generated data, trained models, downloaded CSVs, or credentials.
- Keep ingestion and model fitting in the one-shot CLI process. Start API and
  dashboard only after initialization; API access to DuckDB must be read-only
  and the dashboard uses the API only. Do not introduce simultaneous DuckDB
  writers.
- Preserve ingestion source in persisted metadata; preprocessing must read
  that metadata rather than infer the source from new CLI arguments.
- Keep `delivered` as a canonical boolean from Olist `order_status`; use
  `actual_delivery` only for the retrospective label, never as a feature.
- Demo records and their metrics are illustrative, not evidence about actual
  Brazilian commerce or model performance on Olist.
- Olist inputs are manually downloaded from the official Kaggle dataset.
  Do not redistribute the source CSVs or fetch them implicitly in CI.
- For late-delivery classification, construct features available at order
  purchase time; verify the historical availability of promised date and
  item-level fields before calling an Olist score prospective. For next-week
  demand, use only prior weeks. Use chronological splits and guard against
  label or future-information leakage.
- Update the README, architecture, data dictionary, model cards, and decision
  log when their contracts change. Run `ruff check .`, `pytest -q`, and
  `docker build -t supply-chain-risk-demand-intelligence:ci .` as applicable.

Check the actual package and tests before claiming that commands, endpoints,
or model scores work.
