# Data dictionary

## Input contract

For Olist mode, manually download the **Brazilian E-Commerce Public Dataset
by Olist** from its [official Kaggle page][olist] and put the extracted files
below the chosen input directory (`./input` for Compose). The minimum
expected CSV filenames are:

| File | Relevant source columns | Intended use |
| --- | --- | --- |
| `olist_orders_dataset.csv` | `order_id`, `customer_id`, `order_purchase_timestamp`, `order_estimated_delivery_date`, `order_delivered_customer_date`, `order_status` | Purchase time, retrospective delivery label, customer link, and `delivered = (order_status == "delivered")`; status is not a purchase-time feature. |
| `olist_order_items_dataset.csv` | `order_id`, `product_id`, `price` | Item count, sum of item prices, and primary category; purchase-time availability of item fields needs confirmation. |
| `olist_customers_dataset.csv` | `customer_id`, `customer_state` | Customer link and state feature; raw identifiers are not model features. |
| `olist_products_dataset.csv` | `product_id`, `product_category_name` | Product-to-category mapping for demand; mapping is not a historical catalog snapshot. |
| `product_category_name_translation.csv` | `product_category_name`, `product_category_name_english` | Required CSV supplying the English category; missing translations become `unknown`. |

`order_id` joins orders to items; `customer_id` joins orders to customers;
`product_id` joins items to products; `product_category_name` joins products
to translations. Multiple item rows per order are expected. Parse timestamps
consistently, preserve missing values explicitly, and inspect duplicates and
unmatched joins rather than silently multiplying rows. Olist CSV timestamps
do not carry timezone offsets; a chosen timezone convention must be recorded
with any derived weekly data.

## Derived concepts

| Concept | Definition and boundary |
| --- | --- |
| Order-time cutoff | `order_purchase_timestamp`; candidate classifier features must exist at this instant. |
| Canonical order | One row per `order_id`: purchase / estimated / actual delivery timestamps, `customer_state`, highest-priced item's translated `category` (product ID breaks price ties), `item_count`, summed item `total_price`, and boolean `delivered`. Unknown categories and states are retained as explicit values. Missing checkout items/estimates and impossible date sequences are rejected; the quality report counts rejections. |
| Late-delivery label | Only for `delivered` orders with an actual delivery timestamp, actual delivery **calendar date** after estimated delivery **calendar date**. Same-day arrivals are on time. Missing actual delivery and non-delivered orders are excluded from classifier training. `actual_delivery` and `delivered` are never model features. The recorded estimate's purchase-time provenance is not established. |
| Weekly orders | Count canonical order rows by Monday-start purchase week, primary category, and customer state; this counts orders, not purchased units or shipments. Missing weeks are zero-filled for observed category/state pairs. For Olist, exclude the final (potentially partial) purchase week from demand history; report its excluded order count. Delivery modeling still uses eligible orders from it. |
| Next-week demand target | Order count in the next Monday-start calendar week for a category/state pair; at the forecast cutoff use only earlier weeks for lagged features. Olist timestamps lack timezone offsets; the implementation treats them as naive timestamps. |
| Demo records | Generated illustrative input with no claim of representing Olist or population statistics. |
| Runtime data and artifacts | `PLATFORM_DATA_DIR/platform.duckdb` holds `raw_orders`, `orders`, `demand_history`, `demand_features`, and summary tables. `PLATFORM_ARTIFACT_DIR` holds `ingestion.json` with persisted source, `delivery.joblib`, `demand.joblib`, and JSON model metadata, metrics, and quality reports. |

Do not put the input CSVs in the image, repository, or CI. The Olist dataset
may have missing or inconsistent records; no dataset size, model score, or
production-quality assumption is asserted here.

[olist]: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
