# Methodology

## Data

- Complete FreshRetailNet-50K training split: 4,500,000 daily rows.
- Official evaluation split: 350,000 rows.
- 50,000 store-product series.
- Seven-day forecast horizon.
- No series sampling.

## Forecasting framework

1. Build causal origin-time lags, rolling statistics, identities, and known future
   covariates.
2. Materialize evaluation-like forecast-origin chunks.
3. Tune shared XGBoost hyperparameters with Optuna on all features.
4. Freeze those hyperparameters during individual forward feature selection.
5. Reset temporal chunks and CV after an accepted feature.
6. Stop when the best candidate no longer improves the configured comparison score.
7. Jointly retune the selected set, including L1 and L2 regularization.
8. Fit GPU XGBoost and evaluate once on the untouched official split.

## Experiments

### Mandatory start

Forced initial predictors:

- `horizon`
- `seasonal_naive_7`
- `sales_roll_mean_7`
- `future_discount`
- `future_holiday_flag`

Forward selection then accepted `store_id`.

### Zero start

The initial selected set is empty. A training-mean predictor serves as the
zero-feature baseline. Seven variables were selected from 57 candidates.

## Evaluation metrics

### WAPE

```text
sum(|prediction - target|) / sum(|target|)
```

Lower is better.

### WPE

```text
sum(prediction - target) / sum(target)
```

WPE below zero indicates aggregate underprediction; above zero indicates
overprediction.

### MSE, MAE, and RMSE

These are reported for the repository models but are not available for every paper
baseline in the committed benchmark table.

## Paper-comparable subset

The headline paper comparison uses:

- `subset = uncensored_target_days`;
- `horizon = overall`;
- groups = Overall, Low Sale, High Sale.

All-target-day metrics are also retained in
`results/our_evaluation_all_subsets_and_horizons.csv`.
