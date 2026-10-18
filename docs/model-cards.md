# Model cards

These cards describe modeling intent and known implementation boundaries,
not measured results. Inspect each run's artifact metadata for the selected
estimator, split dates, and observed metrics before publishing predictions.

## Order-time late-delivery classifier

- **Purpose:** estimate the chance that an order eventually delivered to a
  customer arrives after its recorded estimated delivery date. This is a
  retrospective proxy for delivery risk, not a diagnosis of a disruption.
- **Prediction point and inputs:** `order_purchase_timestamp`; implemented
  inputs include purchase month/weekday, customer state, primary category,
  item count, summed item price, and estimated promised days. Olist does not
  prove that the recorded estimate and all item fields were frozen at
  purchase. Verify their availability before claiming prospective
  leakage-safe scoring. Never use eventual status, delivered timestamp,
  post-purchase review, or future aggregates as features.
- **Label and exclusions:** the canonical `delivered` boolean is derived
  from `order_status`; compare actual vs. estimated customer delivery
  timestamps only for delivered orders with both values valid. `delivered`
  and `actual_delivery` are not classifier inputs. Compare calendar dates,
  so delivery later on the promised date is on time. Canceled, missing, and
  still-open outcomes are outside this binary target; assess selection bias
  separately. The dataset does not establish whether its recorded estimate
  was revised after purchase.
- **Evaluation:** implementation uses chronological train/validation/test
  partitions by purchase time and chooses among majority, logistic, and
  random-forest candidates and fixed-grid probability thresholds on validation
  F1; the selected threshold is persisted for inference. It reports test accuracy,
  precision, recall, F1, confusion matrix, and ROC-AUC when defined.
  Historical outcomes may have matured after the training cutoff; a
  production backtest must enforce as-of label maturity and audit calibration,
  class balance, exclusions, and subgroup slices. No scores are asserted here.
- **Limitations:** historical Olist coverage, missing timestamps, potential
  estimate revisions, changing delivery conditions, and selective labels
  constrain generalization. Do not use as an automated shipping decision.

## Next-week category/state order forecast

- **Purpose:** forecast order count per primary product category and customer
  state for the next calendar week. This is not purchased item count,
  inventory demand, fulfilled volume, or a replenishment recommendation.
- **Prediction point and target:** immediately before forecast week `W`,
  estimate orders whose purchase time falls within `W` for that
  category/state pair.
  Lag, rolling, and seasonality inputs use only fully observed weeks before
  `W`. Establish a consistent timezone and week start; do not use the
  forecast week's orders or a full-series normalization fitted on the test
  period. The potentially incomplete final Olist week is excluded from
  demand history; an unseen category/state pair uses zero prior-week lags
  and receives a low-confidence cold-start note.
- **Evaluation:** implementation uses chronological train/validation/test
  periods, a prior-week baseline, and a random forest, selecting by
  validation MAE and reporting test MAE/RMSE. Rolling-origin backtests,
  cold-start coverage, and category/state slices remain necessary for
  broader validation. No accuracy numbers are asserted here.
- **Limitations:** catalog mapping is not timestamped, product mix changes,
  marketplace coverage is historical, and purchases omit latent unmet
  demand. Promotional, stockout, and external disruptions are not observed.

## Demo and governance

Synthetic/demo output only checks that the workflow can run; its metrics do
not validate either model on real-world data. Record dataset provenance,
license/usage constraints, run configuration, training cutoff, validation
method, and model version with each real artifact. Keep raw identifiers and
customer-level data out of public displays where possible. Refresh or
withdraw models when data quality or calibration drifts.
