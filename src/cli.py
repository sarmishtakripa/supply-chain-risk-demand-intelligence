"""Run with python -m src.cli [ingest|preprocess|train-delay|train-demand|evaluate|run-all]."""

import argparse
import hashlib
import logging
from pathlib import Path

import pandas as pd

from src.config import settings
from src.data import clean_orders, demo_orders, olist_orders
from src.features import demand_features, weekly_history
from src.logging_utils import configure_logging
from src.modeling import evaluate, train_delivery, train_demand
from src.storage import load_json, read_table, save_json, write_table

logger = logging.getLogger(__name__)


def ingest(source: str, input_dir: Path | None) -> None:
    if source == "demo":
        raw = demo_orders()
    elif source == "olist":
        if input_dir is None:
            raise ValueError("--input-dir is required for --source olist")
        raw = olist_orders(input_dir)
    else:
        raise ValueError(f"Unsupported source: {source}")
    write_table("raw_orders", raw)
    save_json("ingestion", {"source": source, "input_rows": len(raw)})
    logger.info("Ingested %s raw %s records", len(raw), source)


def preprocess() -> None:
    source = load_json("ingestion")["source"]
    cleaned, report = clean_orders(read_table("raw_orders"))
    report.update({
        "source": source,
        "dataset_kind": "deterministic synthetic demo" if source == "demo" else
        "historical Brazilian Olist",
        "data_fingerprint": hashlib.sha256(
            pd.util.hash_pandas_object(cleaned, index=False).values.tobytes()
        ).hexdigest(),
    })
    demand_orders = cleaned
    if source == "olist":
        latest_week = cleaned["purchased_at"].max().to_period("W-SUN").start_time
        demand_orders = cleaned.loc[cleaned["purchased_at"] < latest_week]
        report["demand_excluded_last_partial_week_orders"] = len(cleaned) - len(demand_orders)
        if demand_orders.empty:
            raise ValueError("No completed Olist purchase week available for demand")
    else:
        report["demand_excluded_last_partial_week_orders"] = 0
    history = weekly_history(demand_orders)
    write_table("orders", cleaned)
    write_table("demand_history", history)
    write_table("demand_features", demand_features(history))
    save_json("quality", report)
    logger.info("Accepted %s orders; rejected %s", len(cleaned), report["rejected_rows"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Supply chain risk and demand pipeline")
    parser.add_argument("command", choices=[
        "ingest", "preprocess", "train-delay", "train-demand", "evaluate", "run-all",
    ])
    parser.add_argument("--source", choices=["demo", "olist"], default=settings.source)
    parser.add_argument("--input-dir", type=Path, help="Directory with original Olist CSV files")
    args = parser.parse_args()
    configure_logging()
    try:
        if args.command in ("ingest", "run-all"):
            ingest(args.source, args.input_dir)
        if args.command in ("preprocess", "run-all"):
            preprocess()
        if args.command in ("train-delay", "run-all"):
            train_delivery()
        if args.command in ("train-demand", "run-all"):
            train_demand()
        if args.command in ("evaluate", "run-all"):
            metrics = evaluate()
            logger.info("Held-out metrics: delivery=%s demand=%s",
                        metrics["delivery"]["test"], metrics["demand"]["test"])
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"Pipeline error: {exc}\n")


if __name__ == "__main__":
    main()
