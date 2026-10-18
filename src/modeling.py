"""Chronological training, baselines, evaluation, and model persistence."""

from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings
from src.features import (
    DELIVERY_CATEGORICAL, DELIVERY_FEATURES, DELIVERY_NUMERIC,
    DEMAND_CATEGORICAL, DEMAND_FEATURES, DEMAND_NUMERIC,
    delivery_features, delivery_labels,
)
from src.storage import load_json, read_table, save_json, save_summary

SCHEMA_VERSION = 1


class LastWeekRegressor(RegressorMixin, BaseEstimator):
    """No-training seasonal baseline that uses the prior week's observed count."""

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "LastWeekRegressor":
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(x["lag_1"], dtype=float)


def splits(frame: pd.DataFrame, date_column: str) -> tuple[pd.DataFrame, ...]:
    dates = np.array(sorted(pd.to_datetime(frame[date_column]).dt.normalize().unique()))
    if len(dates) < 12:
        raise ValueError("At least 12 distinct dates/weeks are needed for chronological splits")
    train_end = dates[int(len(dates) * 0.7)]
    val_end = dates[int(len(dates) * 0.85)]
    dates_in_frame = pd.to_datetime(frame[date_column]).dt.normalize()
    result = (
        frame.loc[dates_in_frame < train_end],
        frame.loc[(dates_in_frame >= train_end) & (dates_in_frame < val_end)],
        frame.loc[dates_in_frame >= val_end],
    )
    if any(part.empty for part in result):
        raise ValueError("Training, validation and test splits must all contain records")
    return result


def preprocessing(categorical: list[str], numeric: list[str], scale: bool = False):
    numeric_steps = [SimpleImputer(strategy="median")]
    if scale:
        numeric_steps.append(StandardScaler())
    return ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", make_pipeline(*numeric_steps), numeric),
    ])


def _metadata(kind: str, model: str, features: list[str], train, val, test) -> dict:
    return {
        "kind": kind, "selected_model": model, "schema_version": SCHEMA_VERSION,
        "sklearn_version": sklearn.__version__, "feature_columns": features,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "train_end": str(pd.to_datetime(train["split_date"]).max().date()),
        "validation_end": str(pd.to_datetime(val["split_date"]).max().date()),
        "test_end": str(pd.to_datetime(test["split_date"]).max().date()),
    }


def _importance(model, x: pd.DataFrame, y: pd.Series, scoring: str) -> dict[str, float]:
    sample = x.iloc[: min(len(x), 600)]
    result = permutation_importance(
        model, sample, y.iloc[: len(sample)], scoring=scoring, n_repeats=2, random_state=41
    )
    return {
        name: round(float(score), 5)
        for name, score in zip(x.columns, result.importances_mean, strict=True)
    }


def _classification_scores(model, x: pd.DataFrame, y: pd.Series,
                           threshold: float = 0.5) -> dict:
    probability = model.predict_proba(x)[:, 1]
    predicted = (probability >= threshold).astype(int)
    scores = {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }
    scores["roc_auc"] = (
        float(roc_auc_score(y, probability))
        if y.nunique() == 2 else None
    )
    return scores


def train_delivery() -> dict:
    orders = read_table("orders")
    orders = orders.loc[orders["delivered"]].dropna(subset=["actual_delivery"]).copy()
    if len(orders) < 50:
        raise ValueError("At least 50 delivered orders are required to train delivery risk")
    orders["split_date"] = pd.to_datetime(orders["purchased_at"])
    train, val, test = splits(orders, "split_date")
    x_train, y_train = delivery_features(train), delivery_labels(train)
    if y_train.nunique() != 2:
        raise ValueError("Training split must contain both on-time and late deliveries")
    x_val, y_val = delivery_features(val), delivery_labels(val)
    x_test, y_test = delivery_features(test), delivery_labels(test)
    candidates = {
        "majority_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(
            preprocessing(DELIVERY_CATEGORICAL, DELIVERY_NUMERIC, scale=True),
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=41),
        ),
        "random_forest": make_pipeline(
            preprocessing(DELIVERY_CATEGORICAL, DELIVERY_NUMERIC),
            RandomForestClassifier(n_estimators=80, min_samples_leaf=5,
                                   class_weight="balanced_subsample", random_state=41, n_jobs=1),
        ),
    }
    validation = {}
    thresholds = {}
    for name, candidate in candidates.items():
        candidate.fit(x_train, y_train)
        options = [0.5] if name == "majority_baseline" else [
            round(value / 100, 2) for value in range(20, 81, 5)
        ]
        threshold, scores = max(
            ((threshold, _classification_scores(candidate, x_val, y_val, threshold))
             for threshold in options),
            key=lambda choice: choice[1]["f1"],
        )
        thresholds[name] = threshold
        validation[name] = scores
    winner = max(candidates, key=lambda name: validation[name]["f1"])
    model = candidates[winner]
    decision_threshold = thresholds[winner]
    metadata = _metadata("delivery", winner, DELIVERY_FEATURES, train, val, test)
    metadata.update({
        "source": load_json("quality")["source"],
        "data_fingerprint": load_json("quality")["data_fingerprint"],
        "decision_threshold": decision_threshold,
        "validation": validation,
        "test": _classification_scores(model, x_test, y_test, decision_threshold),
        "baseline_test": _classification_scores(
            candidates["majority_baseline"], x_test, y_test
        ),
        "test_positive_rate": float(y_test.mean()),
        "permutation_importance_test": _importance(
            model, x_test, y_test,
            lambda estimator, x, y: f1_score(
                y, estimator.predict_proba(x)[:, 1] >= decision_threshold, zero_division=0
            ),
        ),
        "positive_label": "actual delivery calendar date > promised calendar date",
        "eligible_population": "orders with actual delivery date",
    })
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.artifact_dir / "delivery.joblib")
    save_json("delivery_metadata", metadata)
    return metadata


def _regression_scores(model, x: pd.DataFrame, y: pd.Series) -> dict:
    predicted = np.maximum(0, model.predict(x))
    return {
        "mae": float(mean_absolute_error(y, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
    }


def train_demand() -> dict:
    frame = read_table("demand_features").copy()
    frame["split_date"] = pd.to_datetime(frame["week"])
    train, val, test = splits(frame, "split_date")
    x_train, y_train = train[DEMAND_FEATURES], train["orders"]
    x_val, y_val = val[DEMAND_FEATURES], val["orders"]
    x_test, y_test = test[DEMAND_FEATURES], test["orders"]
    candidates = {
        "last_week_baseline": LastWeekRegressor(),
        "random_forest": make_pipeline(
            preprocessing(DEMAND_CATEGORICAL, DEMAND_NUMERIC),
            RandomForestRegressor(n_estimators=80, min_samples_leaf=3, random_state=41,
                                  n_jobs=1),
        ),
    }
    validation = {}
    for name, candidate in candidates.items():
        candidate.fit(x_train, y_train)
        validation[name] = _regression_scores(candidate, x_val, y_val)
    winner = min(candidates, key=lambda name: validation[name]["mae"])
    model = candidates[winner]
    metadata = _metadata("demand", winner, DEMAND_FEATURES, train, val, test)
    metadata.update({
        "source": load_json("quality")["source"],
        "data_fingerprint": load_json("quality")["data_fingerprint"],
        "validation": validation, "test": _regression_scores(model, x_test, y_test),
        "baseline_test": _regression_scores(candidates["last_week_baseline"], x_test, y_test),
        "permutation_importance_test": _importance(model, x_test, y_test,
                                                   "neg_mean_absolute_error"),
        "target": "orders in forecast week by category and customer state",
        "history_end_week": str(pd.to_datetime(frame["week"]).max().date()),
    })
    joblib.dump(model, settings.artifact_dir / "demand.joblib")
    save_json("demand_metadata", metadata)
    return metadata


def evaluate() -> dict:
    delivery = load_json("delivery_metadata")
    demand = load_json("demand_metadata")
    quality = load_json("quality")
    if delivery["schema_version"] != SCHEMA_VERSION or demand["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Model metadata schema is incompatible; retrain both models")
    if (delivery["data_fingerprint"] != quality["data_fingerprint"] or
            demand["data_fingerprint"] != quality["data_fingerprint"]):
        raise ValueError("Models were trained on different curated data; run the full pipeline")
    metrics = {
        "source": quality["source"], "dataset_kind": quality["dataset_kind"],
        "data_fingerprint": quality["data_fingerprint"],
        "delivery": delivery, "demand": demand,
        "warning": "Demo metrics are synthetic, not evidence of real-world performance"
        if quality["source"] == "demo" else
        "Historical Olist holdout results; not prospective or US-market validation",
    }
    save_json("metrics", metrics)
    save_summary(quality, metrics)
    return metrics
