"""Features available at order placement and before each forecast week."""

import pandas as pd

DELIVERY_CATEGORICAL = ["customer_state", "category"]
DELIVERY_NUMERIC = ["purchase_month", "purchase_weekday", "promised_days",
                    "item_count", "total_price"]
DELIVERY_FEATURES = DELIVERY_CATEGORICAL + DELIVERY_NUMERIC
DEMAND_CATEGORICAL = ["category", "customer_state"]
DEMAND_NUMERIC = ["week_of_year", "lag_1", "lag_2", "lag_4", "rolling_4"]
DEMAND_FEATURES = DEMAND_CATEGORICAL + DEMAND_NUMERIC


def delivery_features(orders: pd.DataFrame) -> pd.DataFrame:
    frame = orders.copy()
    purchase = pd.to_datetime(frame["purchased_at"])
    estimated = pd.to_datetime(frame["estimated_delivery"])
    frame["purchase_month"] = purchase.dt.month
    frame["purchase_weekday"] = purchase.dt.dayofweek
    frame["promised_days"] = (
        estimated.dt.normalize() - purchase.dt.normalize()
    ).dt.days
    return frame[DELIVERY_FEATURES]


def delivery_labels(orders: pd.DataFrame) -> pd.Series:
    return (
        pd.to_datetime(orders["actual_delivery"]).dt.normalize() >
        pd.to_datetime(orders["estimated_delivery"]).dt.normalize()
    ).astype(int)


def weekly_history(orders: pd.DataFrame) -> pd.DataFrame:
    frame = orders.copy()
    frame["week"] = pd.to_datetime(frame["purchased_at"]).dt.to_period("W-SUN").dt.start_time
    counts = frame.groupby(["category", "customer_state", "week"]).size().rename("orders")
    start, end = frame["week"].min(), frame["week"].max()
    weeks = pd.date_range(start, end, freq="W-MON")
    pairs = frame[["category", "customer_state"]].drop_duplicates()
    index = pd.MultiIndex.from_product(
        [pairs["category"].unique(), pairs["customer_state"].unique(), weeks],
        names=["category", "customer_state", "week"],
    )
    # Only observed pairs are represented; inference handles previously unseen pairs separately.
    observed = pd.MultiIndex.from_frame(pairs)
    index = index[index.droplevel("week").isin(observed)]
    history = counts.reindex(index, fill_value=0).reset_index()
    history["orders"] = history["orders"].astype(int)
    return history.sort_values(["category", "customer_state", "week"]).reset_index(drop=True)


def demand_features(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.sort_values(["category", "customer_state", "week"]).copy()
    grouped = frame.groupby(["category", "customer_state"], sort=False)["orders"]
    for lag in (1, 2, 4):
        frame[f"lag_{lag}"] = grouped.shift(lag).fillna(0).astype(float)
    frame["rolling_4"] = grouped.transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).mean()
    ).fillna(0.0)
    frame["week_of_year"] = pd.to_datetime(frame["week"]).dt.isocalendar().week.astype(int)
    return frame.reset_index(drop=True)


def next_week_features(history: pd.DataFrame, category: str, state: str) -> tuple[pd.DataFrame, bool]:
    latest = pd.to_datetime(history["week"]).max()
    subset = history.loc[
        (history["category"] == category) & (history["customer_state"] == state)
    ]
    cold_start = subset.empty
    appended = pd.DataFrame([{
        "category": category, "customer_state": state,
        "week": latest + pd.Timedelta(weeks=1), "orders": 0,
    }])
    sequence = (
        subset[["category", "customer_state", "week", "orders"]].copy()
        if not cold_start else appended.iloc[:0].copy()
    )
    features = demand_features(pd.concat([sequence, appended], ignore_index=True))
    return features[DEMAND_FEATURES].tail(1), cold_start
