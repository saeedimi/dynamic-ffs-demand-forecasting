# TimesNet scope: what this repository does and does not reproduce

## What the paper's TimesNet stage does

In the published workflow, observed sales are censored when products are unavailable.
TimesNet is used before forecasting to estimate latent demand during stockout periods.
The recovered signal is then aggregated and supplied to TFT for future forecasting.

Conceptually:

```text
Observed hourly sales
+ hourly stock-status indicators
+ contextual covariates
        |
        v
TimesNet latent-demand recovery
        |
        v
Recovered daily demand target
        |
        v
TFT seven-day forecasting
```

This is a **target-reconstruction stage**, not simply another feature added to a
forecasting model.

## What this repository does

The two experiments train XGBoost directly on raw daily sales:

```text
Raw recorded daily sales
+ causal lags and rolling statistics
+ known future discount/calendar/weather predictors
+ store/product identities
+ recent stockout-history predictors
        |
        v
GPU XGBoost seven-day forecasting
```

The zero-start model selects `stockout_roll_mean_14`, which is meaningful because it
signals recent supply constraints and possible censoring. However, it does not replace
missing latent demand values and does not change the target from recorded sales to
recovered demand.

## Why the distinction matters

A forecasting model trained on raw sales can learn associations with stockouts, but it
cannot observe the counterfactual demand that would have occurred while the item was
unavailable. This is especially consequential for high-sale products:

| Method | High-sale WAPE | High-sale WPE |
|---|---:|---:|
| XGBoost zero start, raw sales | 25.50% | -10.93% |
| Paper TFT + TimesNet recovery | **23.03%** | **+0.87%** |

TimesNet + TFT is both more accurate and much less biased for high-sale products. This
supports the interpretation that explicit latent-demand recovery addresses information
that raw-sales forecasting alone does not fully recover.

## Why zero start can still beat TimesNet + TFT for low-sale WAPE

Zero-start XGBoost reports 36.10% low-sale WAPE versus 37.33% for TFT + TimesNet.
This does not demonstrate superior latent-demand recovery. Possible explanations
include:

- sparse low-sale series may be well served by recent-level and identity features;
- recovery can introduce variance or positive bias when observations are sparse;
- the comparison is made on uncensored evaluation target days, not directly against
  an observed latent-demand ground truth;
- the downstream forecasting architectures and optimization procedures differ.

The TimesNet pipeline has +7.78% low-sale WPE, while zero-start XGBoost has -1.15%.
This is a calibration difference on the evaluation target, not proof that the
XGBoost target is a better estimate of unobserved demand.

## Fair comparison hierarchy

### Primary comparison

Compare this repository with the paper's raw-sales forecasting baselines:

- SSA + raw sales;
- TFT + raw sales;
- DLinear + raw sales.

These methods and the XGBoost experiments share the same broad target type: recorded
sales without TimesNet recovery.

### Secondary comparison

Compare with TFT + TimesNet as an informative two-stage reference. State clearly that:

- its training target has undergone latent-demand reconstruction;
- its architecture is different;
- it uses an additional expensive recovery stage;
- this repository did not successfully reproduce that stage.

## Reproduction status

`timesnet_latent_demand_recovery_reproduced = false`

An attempt to run the official recovery pipeline on a standard 12.7 GB Colab runtime
was terminated before `demand.parquet` was generated. No recovered-demand results are
included in this repository.

## Recommended wording

> We did not reproduce the paper's TimesNet latent-demand recovery. Our XGBoost
> experiments forecast raw recorded sales and are directly comparable to the paper's
> raw-sales baselines. TFT + TimesNet is reported as a separate two-stage reference
> that demonstrates the benefit of explicit demand recovery, particularly for
> high-sale products.
