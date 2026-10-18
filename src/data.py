"""Input adapters and canonical order validation."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "order_id", "purchased_at", "estimated_delivery", "actual_delivery",
    "customer_state", "category", "item_count", "total_price", "delivered",
]
OLIST_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "translations": "product_category_name_translation.csv",
}


def demo_orders(seed: int = 41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = ["electronics", "home", "beauty", "sports"]
    states = ["SP", "RJ", "MG", "BA"]
    rows = []
    start = datetime(2017, 1, 2)
    for week in range(48):
        for category_index, category in enumerate(categories):
            for state_index, state in enumerate(states):
                seasonal = 1.0 + 0.32 * np.sin(week / 5 + category_index)
                count = rng.poisson((2.5 + category_index + (state_index == 0) * 2) * seasonal)
                for n in range(count):
                    purchase = start + timedelta(weeks=week, days=int(rng.integers(0, 7)))
                    price = round(float(rng.uniform(18, 240)), 2)
                    item_count = int(rng.integers(1, 4))
                    promised_days = int(rng.integers(8, 25))
                    promised = purchase + timedelta(days=promised_days)
                    transit = max(
                        1, int(rng.normal(10 + state_index * 2 + category_index, 5))
                    )
                    rows.append(
                        (f"demo-{week:02d}-{category_index}-{state_index}-{n}",
                         purchase, promised, purchase + timedelta(days=transit),
                         state, category, item_count, price * item_count, True)
                    )
    return pd.DataFrame(rows, columns=COLUMNS)


def olist_orders(input_dir: Path) -> pd.DataFrame:
    missing = [name for name in OLIST_FILES.values() if not (input_dir / name).is_file()]
    if missing:
        raise ValueError(f"Missing Olist CSV files in {input_dir}: {', '.join(missing)}")
    orders = pd.read_csv(input_dir / OLIST_FILES["orders"])
    items = pd.read_csv(input_dir / OLIST_FILES["items"])
    customers = pd.read_csv(input_dir / OLIST_FILES["customers"])
    products = pd.read_csv(input_dir / OLIST_FILES["products"])
    translations = pd.read_csv(input_dir / OLIST_FILES["translations"])
    required = {
        "orders": {"order_id", "customer_id", "order_status", "order_purchase_timestamp",
                   "order_estimated_delivery_date", "order_delivered_customer_date"},
        "items": {"order_id", "product_id", "price"},
        "customers": {"customer_id", "customer_state"},
        "products": {"product_id", "product_category_name"},
        "translations": {"product_category_name", "product_category_name_english"},
    }
    for name, frame in (
        ("orders", orders), ("items", items), ("customers", customers),
        ("products", products), ("translations", translations),
    ):
        absent = required[name] - set(frame.columns)
        if absent:
            raise ValueError(f"{OLIST_FILES[name]} missing columns: {sorted(absent)}")
    for name, frame, key in (
        ("customers", customers, "customer_id"),
        ("products", products, "product_id"),
        ("translations", translations, "product_category_name"),
    ):
        if frame[key].duplicated().any():
            raise ValueError(f"{OLIST_FILES[name]} has duplicate {key} values")
    product_categories = products[["product_id", "product_category_name"]].merge(
        translations, on="product_category_name", how="left", validate="many_to_one"
    )
    item_rows = items[["order_id", "product_id", "price"]].merge(
        product_categories[["product_id", "product_category_name_english"]],
        on="product_id", how="left", validate="many_to_one",
    )
    totals = item_rows.groupby("order_id").agg(
        item_count=("product_id", "size"), total_price=("price", "sum")
    )
    primary = (
        item_rows.sort_values(["order_id", "price", "product_id"],
                              ascending=[True, False, True])
        .drop_duplicates("order_id").set_index("order_id")["product_category_name_english"]
    )
    totals["category"] = primary
    merged = orders.merge(
        customers[["customer_id", "customer_state"]], on="customer_id",
        how="left", validate="many_to_one",
    ).merge(totals, on="order_id", how="left", validate="many_to_one")
    merged["delivered"] = merged["order_status"].eq("delivered")
    return merged.rename(columns={
        "order_purchase_timestamp": "purchased_at",
        "order_estimated_delivery_date": "estimated_delivery",
        "order_delivered_customer_date": "actual_delivery",
    })[COLUMNS]


def clean_orders(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = set(COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Canonical orders missing columns: {sorted(missing)}")
    frame = raw[COLUMNS].copy()
    for column in ("purchased_at", "estimated_delivery", "actual_delivery"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ("item_count", "total_price"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not frame["delivered"].dropna().isin([True, False]).all():
        raise ValueError("Canonical delivered values must be boolean")
    frame["delivered"] = frame["delivered"].fillna(False).astype(bool)
    frame["customer_state"] = frame["customer_state"].fillna("UNKNOWN").astype(str).str.strip()
    frame["category"] = frame["category"].fillna("unknown").astype(str).str.strip()
    frame.loc[frame["customer_state"] == "", "customer_state"] = "UNKNOWN"
    frame.loc[frame["category"] == "", "category"] = "unknown"
    duplicate = frame.duplicated("order_id", keep="first")
    invalid = (
        frame["order_id"].isna() | frame["order_id"].astype(str).str.strip().eq("")
        | frame["purchased_at"].isna()
        | frame["estimated_delivery"].isna()
        | (frame["estimated_delivery"].dt.normalize() <
           frame["purchased_at"].dt.normalize())
        | (frame["delivered"] & frame["actual_delivery"].notna() &
           (frame["actual_delivery"] < frame["purchased_at"]))
        | frame["item_count"].isna() | (frame["item_count"] < 1)
        | frame["total_price"].isna() | (frame["total_price"] < 0)
    )
    rejected = duplicate | invalid
    cleaned = frame.loc[~rejected].sort_values(["purchased_at", "order_id"]).reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No valid orders remain after cleaning")
    report = {
        "input_rows": len(frame), "accepted_rows": len(cleaned),
        "rejected_rows": int(rejected.sum()), "duplicate_ids": int(duplicate.sum()),
        "invalid_rows": int(invalid.sum()),
        "missing_actual_delivery": int(cleaned["actual_delivery"].isna().sum()),
        "delivered_with_date": int((cleaned["delivered"] &
                                    cleaned["actual_delivery"].notna()).sum()),
        "unknown_category": int((cleaned["category"] == "unknown").sum()),
        "unknown_state": int((cleaned["customer_state"] == "UNKNOWN").sum()),
        "purchase_start": cleaned["purchased_at"].min().isoformat(),
        "purchase_end": cleaned["purchased_at"].max().isoformat(),
    }
    return cleaned, report
