# Detailed comparison with the FreshRetailNet-50K paper

## Scope and evaluation basis

The comparison uses the official seven-day evaluation and the paper-comparable
`uncensored_target_days` subset:

- 206,562 rows overall;
- 153,208 low-sale rows;
- 53,354 high-sale rows.

The XGBoost experiments forecast raw recorded sales. The paper reports three raw-sales
forecasting baselines (SSA, TFT, and DLinear) and a two-stage TFT result using
TimesNet-recovered latent demand.

## Complete WAPE and WPE comparison

| Group | Method | Target/recovery | WAPE (%) | WPE (%) | Interpretation |
|---|---|---|---:|---:|---|
| Overall | TFT | TimesNet recovery | **29.02** | +2.58 | best overall WAPE; slight overprediction |
| Overall | XGBoost zero start | Raw sales | **29.96** | -6.82 | best raw-sales WAPE; underprediction |
| Overall | DLinear | Raw sales | 31.56 | -4.89 | best paper raw baseline |
| Overall | TFT | Raw sales | 31.75 | -7.37 | similar bias to zero start, higher error |
| Overall | SSA | Raw sales | 31.97 | -10.50 | largest raw-baseline underprediction |
| Overall | XGBoost mandatory start | Raw sales | 33.03 | -5.13 | weakest overall WAPE in this comparison |
| Low Sale | XGBoost zero start | Raw sales | **36.10** | -1.15 | lowest WAPE and near-zero bias |
| Low Sale | TFT | Raw sales | 37.04 | -1.72 | best paper raw baseline |
| Low Sale | TFT | TimesNet recovery | 37.33 | +7.78 | recovery increases positive bias here |
| Low Sale | SSA | Raw sales | 38.33 | -8.03 | strong underprediction |
| Low Sale | XGBoost mandatory start | Raw sales | 38.94 | +3.11 | moderate overprediction |
| Low Sale | DLinear | Raw sales | 39.78 | -1.03 | near-zero bias but highest WAPE |
| High Sale | TFT | TimesNet recovery | **23.03** | **+0.87** | best accuracy and almost unbiased |
| High Sale | XGBoost zero start | Raw sales | **25.50** | -10.93 | best raw-sales WAPE, but substantial underprediction |
| High Sale | DLinear | Raw sales | 25.68 | -7.63 | slightly worse WAPE, less biased |
| High Sale | SSA | Raw sales | 27.40 | -12.26 | strong underprediction |
| High Sale | TFT | Raw sales | 27.95 | -11.46 | strong underprediction |
| High Sale | XGBoost mandatory start | Raw sales | 28.74 | -11.12 | lowest accuracy among high-sale methods |

## Zero-start XGBoost versus every paper method

Positive values below mean XGBoost has higher/worse WAPE; negative values mean it is
better.

| Group | Paper method | Recovery | XGBoost minus paper WAPE (pp) | Relative change |
|---|---|---|---:|---:|
| Overall | SSA | Raw | **-2.01** | -6.30% |
| Overall | TFT | Raw | **-1.79** | -5.65% |
| Overall | DLinear | Raw | **-1.60** | -5.08% |
| Overall | TFT | TimesNet | +0.94 | +3.23% |
| Low Sale | SSA | Raw | **-2.23** | -5.82% |
| Low Sale | TFT | Raw | **-0.94** | -2.54% |
| Low Sale | DLinear | Raw | **-3.68** | -9.25% |
| Low Sale | TFT | TimesNet | **-1.23** | -3.30% |
| High Sale | SSA | Raw | **-1.90** | -6.95% |
| High Sale | TFT | Raw | **-2.45** | -8.78% |
| High Sale | DLinear | Raw | **-0.18** | -0.71% |
| High Sale | TFT | TimesNet | +2.47 | +10.71% |

Exact machine-readable values are in
`results/pairwise_comparison_vs_each_paper_model.csv`.

## Accuracy versus bias

WAPE measures total absolute error relative to total demand. WPE measures signed
aggregate error:

- negative WPE: aggregate underprediction;
- positive WPE: aggregate overprediction;
- WPE near zero: better calibration at the aggregate level.

The zero-start model improves WAPE substantially but does not always improve bias.

### Overall

Zero start reduces WAPE from 33.03% to 29.96% relative to mandatory start, but its WPE
moves from -5.13% to -6.82%. It is more accurate point by point, yet slightly more
underpredicted in aggregate.

Compared with paper raw-sales models, zero start has:

- lower absolute bias than SSA;
- slightly lower absolute bias than raw TFT;
- higher absolute bias than DLinear.

### Low sale

Zero start is both accurate and well calibrated:

- WAPE: 36.10%, best in the entire low-sale comparison;
- WPE: -1.15%, close to zero.

DLinear has similar absolute bias (-1.03%) but much worse WAPE (39.78%), showing why
WAPE and WPE must be discussed together.

### High sale

This is the most important limitation:

- zero-start WAPE is strong at 25.50%;
- WPE is -10.93%, meaning systematic underprediction;
- TFT + TimesNet reaches 23.03% WAPE and +0.87% WPE.

The gap is consistent with the role of latent-demand recovery: high-sale products can
lose more demand during stockouts, and a raw-sales target cannot directly learn sales
that were never observed.

## Mandatory start versus zero start

| Group | WAPE reduction | MSE reduction | MAE reduction | RMSE reduction |
|---|---:|---:|---:|---:|
| Overall | 3.07 pp / 9.30% | 22.33% | 9.30% | 11.87% |
| Low Sale | 2.84 pp / 7.29% | 12.42% | 7.29% | 6.42% |
| High Sale | 3.24 pp / 11.29% | 24.34% | 11.29% | 13.01% |

The improvement appears across multiple error metrics, not only WAPE.

## Forecast-horizon behavior

Zero start improves WAPE at every horizon:

| Horizon | Mandatory WAPE | Zero-start WAPE | Improvement |
|---:|---:|---:|---:|
| 1 | 32.30% | 28.40% | 3.90 pp |
| 2 | 32.58% | 28.87% | 3.72 pp |
| 3 | 32.41% | 29.16% | 3.25 pp |
| 4 | 31.33% | 28.21% | 3.11 pp |
| 5 | 33.70% | 32.21% | 1.50 pp |
| 6 | 34.04% | 30.72% | 3.31 pp |
| 7 | 35.66% | 32.78% | 2.88 pp |

Day 5 remains the hardest horizon and shows the strongest negative WPE in the
zero-start experiment.

## Defensible claims

Use:

> On the full FreshRetailNet-50K official evaluation, zero-start GPU XGBoost
> outperformed all reported raw-sales baselines in WAPE across overall, low-sale, and
> high-sale groups. It did not reproduce TimesNet latent-demand recovery and remained
> behind the paper's TimesNet + TFT pipeline overall and for high-sale products.

Avoid:

- “We reproduced TimesNet.”
- “XGBoost recovered latent demand.”
- “The method is state of the art overall.”
- “Stockout features are equivalent to demand recovery.”
