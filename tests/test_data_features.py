import pandas as pd
import pytest

from src.data import clean_orders, demo_orders, olist_orders
from src.features import (
    delivery_features, delivery_labels, demand_features, next_week_features, weekly_history,
)


def test_demo_is_deterministic_and_reaches_common_cleaner():
    first, report = clean_orders(demo_orders())
    second, _ = clean_orders(demo_orders())
    pd.testing.assert_frame_equal(first, second)
    assert report["accepted_rows"] > 1000
    assert report["rejected_rows"] == 0


def test_order_time_features_ignore_delivery_outcome():
    orders = demo_orders().head(4)
    original = delivery_features(orders)
    orders.loc[:, "actual_delivery"] = pd.Timestamp("2035-01-01")
    pd.testing.assert_frame_equal(original, delivery_features(orders))
    assert "actual_delivery" not in original


def test_same_calendar_day_is_not_late():
    orders = demo_orders().head(1)
    orders.loc[:, "estimated_delivery"] = pd.Timestamp("2017-01-20 00:00:00")
    orders.loc[:, "actual_delivery"] = pd.Timestamp("2017-01-20 17:30:00")
    assert delivery_labels(orders).iloc[0] == 0
    orders.loc[:, "purchased_at"] = pd.Timestamp("2017-01-20 12:00:00")
    cleaned, report = clean_orders(orders)
    assert report["accepted_rows"] == 1
    assert delivery_features(cleaned)["promised_days"].iloc[0] == 0


def test_quality_rejects_duplicates_invalid_rows():
    source = demo_orders().head(2)
    invalid = source.iloc[[1]].copy()
    invalid["order_id"] = "invalid-row"
    invalid["item_count"] = 0
    cleaned, report = clean_orders(pd.concat([source, source.iloc[[0]], invalid]))
    assert len(cleaned) == 2
    assert report["duplicate_ids"] == 1
    assert report["invalid_rows"] == 1


def test_weekly_lags_include_zeros_and_exclude_target_week():
    orders = pd.DataFrame({
        "category": ["a", "a", "a"],
        "customer_state": ["SP"] * 3,
        "purchased_at": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-01-29"]),
    })
    history = weekly_history(orders)
    assert history["orders"].tolist() == [1, 0, 1, 0, 1]
    features = demand_features(history)
    assert features["lag_1"].tolist() == [0, 1, 0, 1, 0]
    assert features["rolling_4"].iloc[2] == pytest.approx(0.5)
    previous = features.iloc[-1][["lag_1", "lag_2", "lag_4", "rolling_4"]].copy()
    altered = history.copy()
    altered.loc[altered.index[-1], "orders"] = 1000
    current = demand_features(altered).iloc[-1][["lag_1", "lag_2", "lag_4", "rolling_4"]]
    pd.testing.assert_series_equal(previous, current)
    prediction, cold = next_week_features(history, "new", "NY")
    assert cold and prediction["lag_1"].iloc[0] == 0
    known, cold = next_week_features(history, "a", "SP")
    assert not cold and known["lag_1"].iloc[0] == 1


def test_olist_adapter_maps_original_csvs(tmp_path):
    pd.DataFrame([{
        "order_id": "1", "customer_id": "c", "order_status": "delivered",
        "order_purchase_timestamp": "2018-01-01",
        "order_estimated_delivery_date": "2018-01-12",
        "order_delivered_customer_date": "2018-01-10",
    }]).to_csv(tmp_path / "olist_orders_dataset.csv", index=False)
    pd.DataFrame([
        {"order_id": "1", "product_id": "p1", "price": 20},
        {"order_id": "1", "product_id": "p2", "price": 40},
    ]).to_csv(tmp_path / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame([{"customer_id": "c", "customer_state": "SP"}]).to_csv(
        tmp_path / "olist_customers_dataset.csv", index=False
    )
    pd.DataFrame([
        {"product_id": "p1", "product_category_name": "one"},
        {"product_id": "p2", "product_category_name": "two"},
    ]).to_csv(tmp_path / "olist_products_dataset.csv", index=False)
    pd.DataFrame([
        {"product_category_name": "one", "product_category_name_english": "first"},
        {"product_category_name": "two", "product_category_name_english": "second"},
    ]).to_csv(tmp_path / "product_category_name_translation.csv", index=False)
    mapped = olist_orders(tmp_path)
    assert mapped.iloc[0]["category"] == "second"
    assert mapped.iloc[0]["item_count"] == 2
    assert mapped.iloc[0]["total_price"] == 60
    cleaned, report = clean_orders(mapped)
    assert len(cleaned) == report["accepted_rows"] == 1
