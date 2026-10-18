# Design decisions

These decisions describe the platform contract and its trade-offs.

## 1. Batch initialization before read-only serving

**Decision:** use `python -m src.cli run-all` as the sole writer. Compose
requires it to finish successfully before API and dashboard startup; data
and artifacts live on shared named volumes, read-only in the serving
containers.

**Rationale:** a single DuckDB writer avoids writer contention and keeps
serving a coherent completed batch. **Trade-off:** refreshes require a
coordinated stop/reinitialize/restart rather than live in-place updates.
The ingested source is persisted and read by preprocessing, so staged CLI
commands cannot silently relabel the origin of stored rows.

## 2. Demo by default; explicit manual Olist ingestion

**Decision:** `PLATFORM_SOURCE=demo` by default. Olist mode uses
`--source olist --input-dir ...`; Compose maps a local `./input` directory
read-only to `/app/input`. No runtime download or bundled dataset.

**Rationale:** a reproducible illustrative start without secrets, and an
explicit boundary for the Kaggle dataset. **Trade-off:** real-data users
must download and place CSVs themselves; demo results cannot establish
real-data model quality.

## 3. Separate API and dashboard

**Decision:** FastAPI runs on port 8000 and Streamlit on 8501. The
dashboard consumes the API through `PLATFORM_API_URL`
(`http://api:8000` in Compose), including `/quality` and `/history`,
rather than opening a DuckDB connection.

**Rationale:** one read-only serving interface and a separate presentation
layer. **Trade-off:** API availability affects the dashboard; Compose
startup order alone does not provide readiness.

## 4. Time-respecting prediction definitions

**Decision:** late-delivery features should be restricted to information
available at purchase time, and next-week category/state order features use
only prior completed weeks. Temporal validation is used, but purchase-time
availability of the Olist promised date and item fields and outcome maturity
at training cutoffs still need verification before prospective claims.

**Rationale:** post-outcome fields and future weekly aggregates create
optimistic leakage. **Trade-off:** fewer candidate features and potentially
smaller labeled training windows. See [model cards](model-cards.md).

## 5. No performance claims without evaluation

**Decision:** document intended metrics and limitations, but publish no
measured scores until real-data experiments and their provenance are
recorded. Demo metrics are illustrative only.

**Rationale:** architecture and sample execution are not evidence of
predictive performance or operational safety.
