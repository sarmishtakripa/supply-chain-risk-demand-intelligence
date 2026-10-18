import json
import os
import subprocess
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.cli import ingest, preprocess
from src.config import settings
from src.data import demo_orders
from src.modeling import evaluate, train_delivery, train_demand


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("platform")
    old_data, old_artifacts = settings.data_dir, settings.artifact_dir
    settings.data_dir, settings.artifact_dir = root / "data", root / "artifacts"
    try:
        ingest("demo", None)
        preprocess()
        delivery = train_delivery()
        demand = train_demand()
        report = evaluate()
        yield delivery, demand, report
    finally:
        settings.data_dir, settings.artifact_dir = old_data, old_artifacts


def test_trained_models_and_api(trained):
    delivery, demand, report = trained
    assert delivery["test_rows"] > 0 and demand["test_rows"] > 0
    assert 0 <= delivery["test"]["f1"] <= 1
    assert demand["test"]["mae"] >= 0
    assert report["source"] == "demo"
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "source": "demo"}
        assert client.get("/models").status_code == 200
        assert client.get("/metrics").json()["dataset_kind"] == "deterministic synthetic demo"
        assert client.get("/quality").json()["rejected_rows"] == 0
        assert client.get("/history").json()
        delivery_response = client.post("/predict/delivery-risk", json={
            "purchased_at": "2017-10-02T12:00:00",
            "estimated_delivery": "2017-10-18T12:00:00",
            "customer_state": "SP", "category": "electronics",
            "item_count": 2, "total_price": 120,
        })
        assert delivery_response.status_code == 200, delivery_response.text
        assert 0 <= delivery_response.json()["late_probability"] <= 1
        assert delivery_response.json()["predicted_late"] == (
            delivery_response.json()["late_probability"] >=
            delivery_response.json()["decision_threshold"]
        )
        demand_response = client.post("/predict/demand", json={
            "category": "electronics", "customer_state": "SP",
        })
        assert demand_response.status_code == 200, demand_response.text
        assert demand_response.json()["forecast_week"] == "2017-12-04"
        assert demand_response.json()["predicted_orders"] >= 0
        unseen = client.post("/predict/demand", json={
            "category": "unseen", "customer_state": "NY",
        })
        assert unseen.status_code == 200
        assert unseen.json()["cold_start"] is True
        assert client.post("/predict/delivery-risk", json={
            "purchased_at": "2017-10-18", "estimated_delivery": "2017-10-02",
            "customer_state": "SP", "category": "electronics",
            "item_count": 1, "total_price": 10,
        }).status_code == 422
        assert client.post("/predict/demand", json={
            "category": "electronics", "customer_state": "",
        }).status_code == 422


def test_olist_csv_adapter_through_full_cli(tmp_path):
    raw = demo_orders()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame({
        "order_id": raw["order_id"], "customer_id": raw["order_id"],
        "order_status": "delivered",
        "order_purchase_timestamp": raw["purchased_at"],
        "order_estimated_delivery_date": raw["estimated_delivery"],
        "order_delivered_customer_date": raw["actual_delivery"],
    }).to_csv(input_dir / "olist_orders_dataset.csv", index=False)
    pd.DataFrame({
        "order_id": raw["order_id"], "product_id": raw["category"],
        "price": raw["total_price"],
    }).to_csv(input_dir / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame({
        "customer_id": raw["order_id"], "customer_state": raw["customer_state"],
    }).to_csv(input_dir / "olist_customers_dataset.csv", index=False)
    categories = sorted(raw["category"].unique())
    pd.DataFrame({"product_id": categories, "product_category_name": categories}).to_csv(
        input_dir / "olist_products_dataset.csv", index=False
    )
    pd.DataFrame({
        "product_category_name": categories,
        "product_category_name_english": categories,
    }).to_csv(input_dir / "product_category_name_translation.csv", index=False)
    artifacts = tmp_path / "artifacts"
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "run-all", "--source", "olist",
         "--input-dir", str(input_dir)],
        env={**os.environ, "PLATFORM_DATA_DIR": str(tmp_path / "data"),
             "PLATFORM_ARTIFACT_DIR": str(artifacts)},
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    quality = json.loads((artifacts / "quality.json").read_text())
    metrics = json.loads((artifacts / "metrics.json").read_text())
    assert quality["source"] == metrics["source"] == "olist"
    assert quality["accepted_rows"] == len(raw)
    assert quality["demand_excluded_last_partial_week_orders"] > 0
    assert metrics["demand"]["history_end_week"] == "2017-11-20"
