# FreshRetailNet-50K Forecasting with Temporal  XGBoost

An independent extension of the **FreshRetailNet-50K** benchmark that studies whether a carefully validated  XGBoost framework can forecast **raw, stockout-censored sales** competitively without reproducing the paper's latent-demand recovery stage.

---

## 1. The problem: observed sales are not always true demand

Retail demand forecasting becomes difficult when products go out of stock. During a stockout, customers may still want to purchase an item, but the transaction cannot occur. The recorded sales value is therefore **censored**: it reflects both customer demand and product availability.

```text
true customer demand
        +
product availability
        ↓
observed sales
```

When a model is trained directly on observed sales, it may incorrectly interpret stockout-related sales drops as weak demand. This can create a feedback loop:

```text
stockout
→ artificially low recorded sales
→ underestimated future demand
→ insufficient replenishment
→ further stockouts
```

This issue is particularly important for fresh retail because products are perishable, demand changes quickly, and inventory decisions must balance stockout risk against waste.

---

## 2. The original FreshRetailNet-50K paper

This project builds on:

> **Wang, Y., Gu, J., Long, L., Li, X., Shen, L., Fu, Z., Zhou, X., and Jiang, X. (2025).**  
> *FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail.*  
> [Paper](https://arxiv.org/abs/2505.16319) · [Dataset](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) · [Official baseline code](https://github.com/Dingdong-Inc/frn-50k-baseline)

The paper introduces a large real-world benchmark designed specifically for studying censored retail demand. The released data contain:

| Property | Value |
|---|---:|
| Store-product time series | 50,000 |
| Stores | 898 |
| Cities | 18 |
| Perishable products | approximately 863–865 SKUs |
| Training period | 90 days |
| Training rows | 4,500,000 |
| Official evaluation rows | 350,000 |
| Forecast horizon | 7 days |
| Stockout prevalence | approximately 20% |

Each daily row includes hierarchical identifiers, normalized sales, hourly sales, hourly stock status, discount information, holidays, promotions, precipitation, temperature, humidity, and wind.

### 2.1 The two research tasks in the paper

The paper separates the problem into two connected tasks.

#### Task A — latent-demand recovery

The goal is to estimate the demand that was not observed during stockout periods. The official implementation includes recovery baselines such as:

- TimesNet
- ImputeFormer
- SAITS
- iTransformer
- GPVAE
- CSDI
- DLinear

The recovery stage uses hourly sales, hourly stock-status annotations, and contextual covariates to reconstruct likely sales during censored hours.

#### Task B — future demand forecasting

After latent demand is recovered, hourly values can be aggregated into recovered daily demand. Forecasting models are then trained to predict the next seven days.

The official forecasting baselines include:

- **SSA** — Similar Scenario Average, a statistical baseline;
- **TFT** — Temporal Fusion Transformer;
- **DLinear** — a linear time-series forecasting architecture.

### 2.2 The paper's two-stage modeling idea

The central paper workflow is:

```text
hourly observed sales
+ hourly stock status
+ contextual covariates
        ↓
latent-demand recovery model
(e.g., TimesNet)
        ↓
recovered hourly demand
        ↓
aggregation to daily demand
        ↓
forecasting model
(e.g., TFT)
        ↓
7-day demand forecast
```

The paper reports that recovering censored demand before forecasting improves prediction accuracy and substantially reduces systematic underestimation. In the reported overall comparison, raw-sales TFT has 31.75% WAPE and -7.37% WPE, while TFT trained after TimesNet recovery reaches 29.02% WAPE and +2.58% WPE.

---

## 3. What this repository investigates

This repository asks a different but complementary question:

> **How competitive can a carefully validated tabular machine-learning framework be when forecasting directly from raw recorded sales?**

Instead of reproducing the paper's recovery stage, this work develops a  XGBoost forecasting framework with:

- causal feature engineering;
- evaluation-aligned temporal cross-validation;
- Bayesian hyperparameter optimization with Optuna;
- individual forward feature selection;
- fold and chunk resets after accepted features;
- final joint optimization including L1 and L2 regularization;
- evaluation on the untouched official seven-day split.

This is an **independent forecasting extension**, not a reimplementation of the full paper pipeline.

---

## 4. Our forecasting framework

```text
raw daily sales
+ lag and rolling statistics
+ stockout-history features
+ known future covariates
+ store/product/context features
        ↓
initial Optuna tuning on all candidate features
        ↓
freeze core XGBoost hyperparameters
        ↓
individual forward feature selection
using evaluation-like temporal folds
        ↓
reset temporal chunks and CV after each accepted feature
        ↓
final Optuna tuning on selected features
including L1 and L2 regularization
        ↓
 XGBoost
        ↓
official 7-day evaluation
```

### 4.1 Evaluation-aligned temporal validation

The official task forecasts seven consecutive future days from a single origin. To better match that deployment setting, each development fold uses:

```text
training: all eligible earlier forecast-origin chunks
validation: one complete origin × horizons 1–7 × all 50,000 series
```

This avoids random row-level splitting, which could mix nearby dates or related observations across training and validation.

### 4.2 Two feature-selection variants

#### Mandatory-start experiment

Five predictors are imposed before forward selection:

```text
horizon
seasonal_naive_7
sales_roll_mean_7
future_discount
future_holiday_flag
```

Forward selection then adds features only when the validation score improves.

#### Zero-start experiment

The selected set begins empty. The initial reference is an intercept-only training-mean predictor, and all 57 individual candidate features must earn inclusion.

Starting from zero does **not** mean the final model has no predictors. It is an ablation design that avoids forcing assumptions into the feature set.

### 4.3 Selected zero-start features

| Order | Feature | Interpretation |
|---:|---|---|
| 1 | `sales_roll_mean_7` | recent weekly sales level |
| 2 | `store_id` | persistent store-specific effects |
| 3 | `rain_x_holiday` | interaction between precipitation and holiday context |
| 4 | `product_id` | product-specific demand scale and pattern |
| 5 | `future_discount` | known future promotion intensity |
| 6 | `stockout_roll_mean_14` | recent stockout and censoring pressure |
| 7 | `sales_origin` | most recent observed sales state |

These features are meaningful for forecasting previously observed store-product series. However, `store_id` and `product_id` may capture identity-specific effects and therefore do not establish generalization to completely new stores or products.

---

## 5. Important scope: TimesNet recovery was not reproduced

The distinction between the paper and this repository is central.

### Original paper comparison pipeline

```text
censored hourly sales
→ TimesNet latent-demand recovery
→ recovered daily demand
→ TFT forecasting
```

### This repository

```text
censored raw daily sales
→ engineered predictors
→  XGBoost forecasting
```

`stockout_roll_mean_14` informs XGBoost that recent observed sales may have been affected by availability. It does **not** estimate the unobserved transactions that would have occurred during stockouts. The target remains raw recorded sales.

Therefore:

1. The most direct comparison is with the paper's **raw-sales SSA, TFT, and DLinear** baselines.
2. **TFT + TimesNet** is included as a secondary reference showing what explicit latent-demand recovery can achieve.
3. This repository does not claim to reproduce, replace, or validate TimesNet.
4. A lower WAPE than TFT + TimesNet in one subgroup does not prove better latent-demand recovery because the targets, training pipelines, and model architectures differ.

See [`docs/TIMESNET_SCOPE.md`](docs/TIMESNET_SCOPE.md) for the detailed scope discussion.

---

## 6. Results

Paper-comparable results below use the official seven-day evaluation and the paper's uncensored-target-day subset.

| Sales group | Mandatory-start XGBoost | Zero-start XGBoost | Best paper raw-sales model | Paper TFT + TimesNet |
|---|---:|---:|---:|---:|
| **Overall WAPE** | 33.03% | **29.96%** | DLinear: 31.56% | **29.02%** |
| **Low-sale WAPE** | 38.94% | **36.10%** | TFT: 37.04% | 37.33% |
| **High-sale WAPE** | 28.74% | **25.50%** | DLinear: 25.68% | **23.03%** |
| **Overall WPE** | -5.13% | -6.82% | DLinear: -4.89% | +2.58% |
| **Low-sale WPE** | +3.11% | **-1.15%** | DLinear: -1.03% | +7.78% |
| **High-sale WPE** | -11.12% | -10.93% | DLinear: -7.63% | **+0.87%** |

### 6.1 Overall performance

Zero-start XGBoost achieves 29.96% overall WAPE:

- 1.60 percentage points better than raw-sales DLinear;
- 1.79 points better than raw-sales TFT;
- 2.01 points better than raw-sales SSA;
- 0.94 points behind TFT + TimesNet.

This supports the conclusion that structured temporal validation and feature selection can make a tabular raw-sales model competitive with the paper's deep-learning baselines.

### 6.2 Low-sale products

Zero-start XGBoost achieves the lowest low-sale WAPE in the comparison at 36.10%:

- 0.94 points better than raw-sales TFT;
- 2.23 points better than raw-sales SSA;
- 3.68 points better than raw-sales DLinear;
- 1.23 points better than TFT + TimesNet.

Its low-sale WPE is -1.15%, indicating relatively small aggregate bias.

### 6.3 High-sale products

Zero-start XGBoost reaches 25.50% high-sale WAPE, narrowly improving on raw-sales DLinear at 25.68%. However, it remains 2.47 points behind TFT + TimesNet.

The more important difference is bias:

```text
Zero-start XGBoost: WPE = -10.93%
TFT + TimesNet:     WPE =  +0.87%
```

This suggests that direct raw-sales forecasting still inherits substantial stockout-related underprediction for high-demand products. The paper's explicit recovery stage appears especially valuable in this group.

### 6.4 Mandatory start versus zero start

Zero start improves WAPE for all three groups:

| Group | Mandatory start | Zero start | Improvement |
|---|---:|---:|---:|
| Overall | 33.03% | **29.96%** | 3.07 points |
| Low sale | 38.94% | **36.10%** | 2.84 points |
| High sale | 28.74% | **25.50%** | 3.24 points |

The improvement also appears in MSE, MAE, and RMSE, indicating that the result is not limited to a single evaluation metric.

The full comparison is available in [`docs/PAPER_COMPARISON.md`](docs/PAPER_COMPARISON.md).

![Overall WAPE comparison](figures/wape_overall.png)

---

## 7. Evaluation metrics

### Weighted Absolute Percentage Error

```text
WAPE = Σ|y - ŷ| / Σ|y|
```

Lower WAPE is better. It measures total absolute forecasting error relative to total observed demand.

### Weighted Percentage Error

```text
WPE = Σ(ŷ - y) / Σy
```

WPE measures aggregate directional bias:

- negative WPE: systematic underprediction;
- positive WPE: systematic overprediction;
- WPE near zero: approximately unbiased aggregate forecasts.

MSE, MAE, and RMSE are also reported for the two XGBoost experiments.

---

## 8. Methodological limitations

- TimesNet latent-demand recovery was not reproduced.
- The model target remains stockout-censored raw sales.
- `store_id` and `product_id` may capture memorized identity effects.
- The executed feature-selection workflow resets temporal folds after accepted features. Candidate comparisons within a step use common folds, but reported improvements across different resets are not perfectly paired.
- A `same_reset_baseline` configuration is included for a stronger confirmatory experiment.
- Results are from one completed experimental seed rather than repeated-seed confidence intervals.
- The full search is computationally expensive because it uses all 50,000 series, multiple folds, 57 feature candidates, and two Optuna stages.

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

---

## 9. Repository structure

```text
notebooks/
  01_mandatory_start.ipynb
  02_zero_start.ipynb
  03_paper_comparison.ipynb
  executed/                    completed runs with outputs

results/
  paper_benchmark_table3.csv
  full_benchmark_ranking.csv
  pairwise_comparison_vs_each_paper_model.csv
  our_paper_comparable_summary.csv
  our_evaluation_all_subsets_and_horizons.csv
  horizon_comparison_uncensored_overall.csv
  feature_selection_path.csv
  selected_features_and_interpretation.csv
  hyperparameters.csv
  run_manifest.json

docs/
  METHODOLOGY.md
  PAPER_COMPARISON.md
  TIMESNET_SCOPE.md
  FEATURES.md
  LIMITATIONS.md
  REPRODUCIBILITY.md

figures/
scripts/
src/
tests/
configs/
```

---

## 10. Reproduction

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the test suite and regenerate the figures:

```bash
pytest
python scripts/build_figures.py
```

For full model training, use a GPU-enabled Colab runtime and run one of the experiment notebooks from top to bottom:

```text
notebooks/01_mandatory_start.ipynb
notebooks/02_zero_start.ipynb
```

The dataset is downloaded directly from Hugging Face and is not redistributed in this repository.

---

## 11. Data, code, and citation

### Original resources

- **Paper:** [FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail](https://arxiv.org/abs/2505.16319)
- **Dataset:** [Dingdong-Inc/FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
- **Official baseline:** [Dingdong-Inc/frn-50k-baseline](https://github.com/Dingdong-Inc/frn-50k-baseline)

### Paper citation

```bibtex
@article{2025freshretailnet50k,
  title={FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail},
  author={Wang, Yangyang and Gu, Jiawei and Long, Li and Li, Xin and Shen, Li and Fu, Zhouyu and Zhou, Xiangjun and Jiang, Xu},
  year={2025},
  eprint={2505.16319},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.16319}
}
```

The FreshRetailNet-50K dataset is released under **CC BY 4.0**. The official baseline repository is released under **Apache-2.0**. This repository is an independent extension focused on  XGBoost, temporal validation, Optuna optimization, and forward-feature-selection ablations.
