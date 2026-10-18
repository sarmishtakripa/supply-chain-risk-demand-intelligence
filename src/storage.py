"""Single-writer pipeline persistence; serving processes read only."""

import json
from pathlib import Path

import duckdb
import pandas as pd

from src.config import settings


def write_table(name: str, frame: pd.DataFrame) -> None:
    if name not in {"raw_orders", "orders", "demand_history", "demand_features"}:
        raise ValueError(f"Unsupported table: {name}")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(settings.db_path)) as connection:
        connection.register("incoming", frame)
        connection.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM incoming")
        connection.unregister("incoming")


def read_table(name: str) -> pd.DataFrame:
    if name not in {"raw_orders", "orders", "demand_history", "demand_features"}:
        raise ValueError(f"Unsupported table: {name}")
    if not settings.db_path.is_file():
        raise FileNotFoundError(f"Database not found: {settings.db_path}; run the pipeline first")
    with duckdb.connect(str(settings.db_path), read_only=True) as connection:
        try:
            return connection.execute(f"SELECT * FROM {name}").df()
        except duckdb.CatalogException as exc:
            raise RuntimeError(f"Missing table {name}; run pipeline prerequisites") from exc


def save_json(name: str, payload: dict) -> None:
    if name not in {"ingestion", "quality", "metrics", "delivery_metadata", "demand_metadata"}:
        raise ValueError(f"Unsupported artifact: {name}")
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    path = settings.artifact_dir / f"{name}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_json(name: str) -> dict:
    path = Path(settings.artifact_dir) / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run python -m src.cli run-all")
    return json.loads(path.read_text())


def save_summary(quality: dict, metrics: dict) -> None:
    with duckdb.connect(str(settings.db_path)) as connection:
        connection.execute(
            "CREATE OR REPLACE TABLE quality_summary AS SELECT ?::VARCHAR AS report",
            [json.dumps(quality)],
        )
        connection.execute(
            "CREATE OR REPLACE TABLE model_metrics AS SELECT ?::VARCHAR AS report",
            [json.dumps(metrics)],
        )
