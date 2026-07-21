# Feature interpretation

## Zero-start selected set

| Order | Feature | Why it is meaningful | Main limitation |
|---:|---|---|---|
| 1 | `sales_roll_mean_7` | Captures recent weekly demand level and smooths daily noise. | Must use only observations available before the origin. |
| 2 | `store_id` | Encodes persistent store scale, operations, neighborhood, and assortment effects. | Supports seen stores; not evidence of new-store generalization. |
| 3 | `rain_x_holiday` | Rain may affect trips and delivery differently on holidays than normal days. | Sparse interaction; future weather must be available or forecast. |
| 4 | `product_id` | Encodes product-specific demand scale, category, and purchasing pattern. | Supports seen products; not evidence of new-product generalization. |
| 5 | `future_discount` | Planned promotion intensity is a direct known driver of demand. | Requires promotion schedules at prediction time. |
| 6 | `stockout_roll_mean_14` | Signals recent availability constraints and censored-sales risk. | Mixes customer demand with replenishment policy and does not recover latent demand. |
| 7 | `sales_origin` | Latest observed state provides a strong local level anchor. | Must be computed exactly at the forecast origin. |

## Why the set is coherent

The selected variables cover five complementary information types:

1. recent demand state;
2. store and product heterogeneity;
3. known future promotion;
4. stockout history;
5. contextual weather/calendar interaction.

The selection is parsimonious: seven variables out of 57 candidates.

## Why unselected features may still be meaningful

A feature not selected is not necessarily unimportant scientifically. Forward
selection asks whether a candidate improves the chosen score given the variables
already selected, the fixed hyperparameters, and the sampled temporal folds.

For example:

- `future_holiday_flag` may be partly represented through `rain_x_holiday`;
- `seasonal_naive_7` may be redundant with `sales_roll_mean_7` and `sales_origin`;
- `horizon` may not add enough after the model learns from other predictors;
- a rejected lag may be informative but too correlated with recent rolling demand.

## Identity-feature caveat

`store_id` and `product_id` are meaningful for the benchmark because the official
evaluation contains known series. They can let tree models learn persistent offsets
and interactions. They also create a memorization pathway.

A future robustness extension should report:

- store-held-out validation;
- product-held-out validation;
- category-only models without product identity;
- performance on infrequent or newly introduced series.

## Zero-start validity

Starting with zero features is statistically reasonable when the reference is an
intercept-only training-mean predictor. It provides a transparent lower bound and
forces the first selected feature to beat that reference.

It is best presented as a feature-selection ablation, not as a claim that domain
knowledge is unnecessary.
