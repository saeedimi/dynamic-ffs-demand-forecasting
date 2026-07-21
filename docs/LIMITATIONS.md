# Limitations

## TimesNet was not reproduced

This repository does not contain recovered demand, a TimesNet checkpoint, or a
successful execution of the paper's latent-demand recovery pipeline. The XGBoost
target is raw recorded sales.

## The TimesNet comparison is not apples-to-apples

TFT + TimesNet has an additional target-reconstruction stage and a different
forecasting architecture. It is included to show the best published two-stage
reference, not as a direct raw-sales baseline.

## Feature-selection acceptance across resets

Both completed runs use `feature_selection_acceptance_rule = uploaded_notebook`.
Candidates within a step share the same folds, but after an accepted feature the
chunks and CV are reset. The new best candidate score can then be compared with an
accepted score from an earlier reset.

Consequently, reported improvements across selection steps are not perfectly paired.
A confirmatory run should evaluate the already-selected set and every candidate on
the same newly sampled reset before deciding whether to accept the next feature.

## No repeated-seed uncertainty intervals

The official evaluation is a single holdout. The repository includes one fresh
four-fold temporal assessment per final model, but the folds are not identical across
the two runs. Formal confidence intervals or repeated-seed tests are not available.

## Identity-feature generalization

`store_id` and `product_id` are effective for known-series forecasting. Results do
not establish performance for new stores, new products, or unseen categories.

## Raw-sales target remains censored

Stockout features describe recent availability, but they do not reveal sales that
could not occur. High-sale WPE around -11% shows the remaining underprediction risk.

## Hyperparameter and feature-selection coupling

Initial hyperparameters are optimized using all features and then frozen during
selection. A different initial search or nested selection/tuning process may choose a
different feature set.

## Compute cost

The full procedure uses GPU acceleration but remains expensive due to 50,000 series,
many candidate fits, temporal folds, and 300 final Optuna trials.
