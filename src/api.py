"""Read-only prediction API. Pipeline is the only DuckDB writer."""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.config import settings
from src.features import (
    DELIVERY_FEATURES, DEMAND_FEATURES, delivery_features, next_week_features,
)
from src.modeling import SCHEMA_VERSION
from src.storage import load_json, read_table


class DeliveryRequest(BaseModel):
    purchased_at: datetime
    estimated_delivery: datetime
    customer_state: Annotated[str, Field(min_length=2, max_length=32)]
    category: Annotated[str, Field(min_length=1, max_length=120)]
    item_count: Annotated[int, Field(ge=1)]
    total_price: Annotated[float, Field(ge=0)]

    @model_validator(mode="after")
    def validate_dates(self):
        purchase = pd.Timestamp(self.purchased_at)
        promised = pd.Timestamp(self.estimated_delivery)
        if purchase.tzinfo is not None or promised.tzinfo is not None:
            raise ValueError("Use timezone-naive timestamps as in the Olist CSVs")
        if promised.normalize() < purchase.normalize():
            raise ValueError("estimated_delivery date must not precede purchase date")
        return self


class DeliveryResponse(BaseModel):
    late_probability: float
    predicted_late: bool
    decision_threshold: float
    model: str
    source: str
    population_note: str


class DemandRequest(BaseModel):
    category: Annotated[str, Field(min_length=1, max_length=120)]
    customer_state: Annotated[str, Field(min_length=2, max_length=32)]


class DemandResponse(BaseModel):
    forecast_week: str
    predicted_orders: float
    cold_start: bool
    model: str
    source: str
    note: str


class ModelStore:
    def __init__(self):
        self.delivery_meta = load_json("delivery_metadata")
        self.demand_meta = load_json("demand_metadata")
        self.metrics = load_json("metrics")
        self.quality = load_json("quality")
        for kind, meta, features in (
            ("delivery", self.delivery_meta, DELIVERY_FEATURES),
            ("demand", self.demand_meta, DEMAND_FEATURES),
        ):
            if (meta["schema_version"] != SCHEMA_VERSION or
                    meta["sklearn_version"] != sklearn.__version__ or
                    meta["feature_columns"] != features):
                raise RuntimeError(f"{kind} artifact incompatible with current application")
            if meta["data_fingerprint"] != self.quality["data_fingerprint"]:
                raise RuntimeError(f"{kind} artifact differs from curated data")
        if self.metrics["data_fingerprint"] != self.quality["data_fingerprint"]:
            raise RuntimeError("Metrics differ from curated data")
        self.delivery = joblib.load(settings.artifact_dir / "delivery.joblib")
        self.demand = joblib.load(settings.artifact_dir / "demand.joblib")
        self.history = read_table("demand_history")
        if (str(pd.to_datetime(self.history["week"]).max().date()) !=
                self.demand_meta["history_end_week"]):
            raise RuntimeError("Demand model and curated history have different date ranges")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = ModelStore()
    yield


app = FastAPI(title="Supply Chain Risk & Demand Intelligence", version="0.1.0",
              lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "source": app.state.models.quality["source"]}


@app.get("/models")
def models() -> dict:
    store = app.state.models
    return {
        "source": store.quality["source"], "dataset_kind": store.quality["dataset_kind"],
        "delivery": store.delivery_meta, "demand": store.demand_meta,
    }


@app.get("/metrics")
def metrics() -> dict:
    return app.state.models.metrics


@app.get("/quality")
def quality() -> dict:
    return app.state.models.quality


@app.get("/history")
def history() -> list[dict]:
    frame = app.state.models.history.copy()
    frame["week"] = pd.to_datetime(frame["week"]).dt.strftime("%Y-%m-%d")
    return frame.to_dict(orient="records")


@app.post("/predict/delivery-risk", response_model=DeliveryResponse)
def predict_delivery(request: DeliveryRequest) -> DeliveryResponse:
    store = app.state.models
    frame = pd.DataFrame([request.model_dump()])
    try:
        features = delivery_features(frame)
        probability = float(store.delivery.predict_proba(features)[0, 1])
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid delivery features: {exc}") from exc
    return DeliveryResponse(
        late_probability=probability,
        predicted_late=probability >= store.delivery_meta["decision_threshold"],
        decision_threshold=store.delivery_meta["decision_threshold"],
        model=store.delivery_meta["selected_model"], source=store.quality["source"],
        population_note="Estimated for orders eventually delivered; not a cancellation predictor",
    )


@app.post("/predict/demand", response_model=DemandResponse)
def predict_demand(request: DemandRequest) -> DemandResponse:
    store = app.state.models
    try:
        features, cold_start = next_week_features(
            store.history, request.category, request.customer_state
        )
        prediction = max(0.0, float(store.demand.predict(features)[0]))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid demand features: {exc}") from exc
    next_week = pd.to_datetime(store.history["week"]).max() + pd.Timedelta(weeks=1)
    return DemandResponse(
        forecast_week=next_week.strftime("%Y-%m-%d"),
        predicted_orders=round(prediction, 2), cold_start=cold_start,
        model=store.demand_meta["selected_model"], source=store.quality["source"],
        note="Unseen pair: historical lags are zero; forecast is low-confidence"
        if cold_start else "Next calendar week's category/state order count",
    )
