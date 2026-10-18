"""Streamlit multipage frontend; all runtime data comes from the read-only API."""

from datetime import datetime, time, timedelta

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src.config import settings

st.set_page_config(page_title="Supply Chain Intelligence", layout="wide")


def api(method: str, endpoint: str, payload: dict | None = None):
    try:
        response = requests.request(
            method, f"{settings.api_url}{endpoint}", json=payload, timeout=15
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API request failed ({endpoint}): {exc}")
        st.stop()


@st.cache_data(ttl=60)
def snapshot():
    return api("GET", "/metrics"), api("GET", "/quality"), pd.DataFrame(api("GET", "/history"))


def header():
    metrics, quality, history = snapshot()
    st.title("Supply Chain Risk & Demand Intelligence")
    if quality["source"] == "demo":
        st.warning("DEMO DATA — deterministic synthetic orders. Metrics are not real-world evidence.")
    else:
        st.info("Historical Brazilian Olist data — not a prospective or US-market validation.")
    return metrics, quality, history


def overview():
    metrics, quality, history = header()
    a, b, c = st.columns(3)
    a.metric("Accepted orders", f"{quality['accepted_rows']:,}")
    b.metric("Delivery holdout F1", f"{metrics['delivery']['test']['f1']:.3f}")
    c.metric("Demand holdout MAE", f"{metrics['demand']['test']['mae']:.2f}")
    history["week"] = pd.to_datetime(history["week"])
    weekly = history.groupby("week", as_index=False)["orders"].sum()
    st.plotly_chart(px.line(weekly, x="week", y="orders", title="Weekly order volume"),
                    use_container_width=True)
    st.caption("Only data available at order placement or before a forecast week enters a model.")


def delivery_predictor():
    _, _, history = header()
    st.subheader("Order-placement delivery risk")
    st.caption("Estimate late delivery among orders that will eventually be delivered.")
    with st.form("delivery"):
        a, b = st.columns(2)
        purchase = a.date_input("Purchase date", value=datetime(2017, 10, 2))
        promised = b.date_input("Promised delivery date", value=datetime(2017, 10, 18))
        state = a.selectbox("Customer state", sorted(history["customer_state"].unique()))
        category = b.selectbox("Primary checkout category", sorted(history["category"].unique()))
        count = a.number_input("Item count", min_value=1, value=2)
        price = b.number_input("Item price total", min_value=0.0, value=120.0, step=1.0)
        submitted = st.form_submit_button("Predict delivery risk")
    if submitted:
        result = api("POST", "/predict/delivery-risk", {
            "purchased_at": datetime.combine(purchase, time(12)).isoformat(),
            "estimated_delivery": datetime.combine(promised, time(12)).isoformat(),
            "customer_state": state, "category": category,
            "item_count": int(count), "total_price": float(price),
        })
        st.metric("Late-delivery probability", f"{result['late_probability']:.1%}")
        st.write(f"Classification: {'Late' if result['predicted_late'] else 'On time'} "
                 f"· Model: {result['model']}")
        st.caption(
            result["population_note"] +
            f" · Decision threshold: {result['decision_threshold']:.2f}"
        )


def demand_forecast():
    metrics, _, history = header()
    st.subheader("Next-week demand forecast")
    a, b = st.columns(2)
    category = a.selectbox("Product category", sorted(history["category"].unique()))
    state = b.selectbox("Customer state", sorted(history["customer_state"].unique()))
    if st.button("Forecast next week"):
        result = api("POST", "/predict/demand",
                     {"category": category, "customer_state": state})
        st.metric(f"Predicted orders · week of {result['forecast_week']}",
                  f"{result['predicted_orders']:.2f}")
        st.caption(result["note"] + f" · Model: {result['model']}")
    filtered = history.loc[
        (history["category"] == category) & (history["customer_state"] == state)
    ].copy()
    filtered["week"] = pd.to_datetime(filtered["week"])
    st.plotly_chart(px.bar(filtered, x="week", y="orders", title="Observed weekly demand"),
                    use_container_width=True)
    st.caption("Forecast week begins " +
               str((pd.to_datetime(metrics["demand"]["history_end_week"]) +
                    timedelta(weeks=1)).date()) + ".")


def performance():
    metrics, _, _ = header()
    for kind, label in (("delivery", "Delivery classification"),
                        ("demand", "Weekly demand regression")):
        model = metrics[kind]
        st.subheader(label)
        st.write(f"Selected: **{model['selected_model']}** · Train/validation/test: "
                 f"{model['train_rows']}/{model['validation_rows']}/{model['test_rows']} "
                 f"· Test ends {model['test_end']}")
        st.write("Held-out test:", model["test"])
        st.write("Held-out baseline:", model["baseline_test"])
        st.write("Validation candidates:", model["validation"])
        importance = pd.DataFrame([
            {"feature": key, "score": score}
            for key, score in model["permutation_importance_test"].items()
        ])
        st.plotly_chart(
            px.bar(importance.sort_values("score"), x="score", y="feature",
                   orientation="h", title="Held-out permutation importance"),
            use_container_width=True,
        )
        if kind == "delivery":
            confusion = pd.DataFrame(model["test"]["confusion_matrix"],
                                     index=["Actual on time", "Actual late"],
                                     columns=["Predicted on time", "Predicted late"])
            st.dataframe(confusion)
    st.caption("Permutation importance is a model sensitivity estimate, not a causal explanation.")


def data_quality():
    _, quality, history = header()
    st.subheader("Ingestion and data quality")
    st.json(quality)
    st.write("Zero-demand weeks in the curated panel:",
             int((history["orders"] == 0).sum()), "of", len(history))
    st.caption("Invalid rows and duplicate IDs are rejected; missing categories/states are explicit.")


def architecture():
    header()
    st.subheader("How it works")
    st.markdown("""
    **Adapters** generate deterministic demo rows or join manually downloaded Olist CSVs.
    A common cleaner validates canonical orders and writes DuckDB tables.
    Chronological train/validation/test splits produce persisted scikit-learn pipelines.
    The API loads version-checked artifacts on startup and reads DuckDB without writing;
    this dashboard calls the API.

    **Delivery:** purchase date, promised lead time, checkout category/items/price and
    customer state only. Delivered-order selection creates survivor bias.

    **Demand:** one-week-ahead category × customer-state counts; lag and rolling
    features use preceding weeks only. New combinations use zero historical lags
    and should be treated as low-confidence forecasts.

    **Deployment:** Docker Compose uses a one-shot initializer. AWS services are
    documented as an optional conceptual mapping, not deployed.
    """)


navigation = st.navigation([
    st.Page(overview, title="Overview", icon="📊"),
    st.Page(delivery_predictor, title="Delivery predictor", icon="🚚"),
    st.Page(demand_forecast, title="Demand forecast", icon="📈"),
    st.Page(performance, title="Model performance", icon="🧪"),
    st.Page(data_quality, title="Data quality", icon="🔎"),
    st.Page(architecture, title="About & architecture", icon="🏗️"),
])
navigation.run()
